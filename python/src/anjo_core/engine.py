"""Async end-to-end orchestration over injected model, store, and retriever adapters."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from .affect import TurnShapePolicy, decoding_params, turn_shape_directive
from .appraisal import AppraisalPolicyInput, AppraisalResult, default_appraisal_policy
from .models import (
    CognitionState,
    CompanionState,
    GateInput,
    GateResult,
    GenerateInput,
    MemoryCandidate,
    Message,
    PresenceVector,
    RetrievalInput,
    TurnResult,
)
from .prompt import PromptInputs, PromptPolicy, build_system_prompt, build_untrusted_context
from .protocols import AppraisalPolicy, MemoryRetriever, ModelAdapter, StateStore
from .retrieval import rank_candidates
from .surfacing import build_presence_vector

DEFAULT_GATE_RESULT = GateResult(intent="CASUAL", should_respond=True, should_retrieve=False)
SILENT_GATE_RESULT = GateResult(intent="CASUAL", should_respond=False, should_retrieve=False)
BUILTIN_INTENTS = frozenset(
    {"ABUSE", "APOLOGY", "VULNERABILITY", "CURIOSITY", "CHALLENGE", "NEGLECT", "CASUAL"}
)
GateErrorMode = Literal["raise", "respond", "silent"]
_INTENT_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class EngineLimits:
    """Positive resource ceilings applied by :class:`CompanionEngine`."""

    max_message_chars: int = 16_000
    max_history_messages: int = 200
    max_history_chars: int = 128_000
    max_prompt_chars: int = 128_000
    max_output_chars: int = 32_000

    def __post_init__(self) -> None:
        for name in (
            "max_message_chars",
            "max_history_messages",
            "max_history_chars",
            "max_prompt_chars",
            "max_output_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


DEFAULT_ENGINE_LIMITS = EngineLimits()


def normalize_intent(intent: str, *, custom_intents: Iterable[str] = ()) -> str:
    """Normalize a gate label and reject unregistered custom behavior.

    Built-in labels drive deterministic appraisal. Registered custom labels are
    passed through in normalized form and intentionally receive no built-in
    appraisal impulse.
    """
    if not isinstance(intent, str):
        raise TypeError("gate intent must be a string")
    normalized = intent.strip().upper()
    allowed_custom = frozenset(_normalize_custom_intent(item) for item in custom_intents)
    if normalized not in BUILTIN_INTENTS and normalized not in allowed_custom:
        raise ValueError(f"unsupported gate intent: {intent!r}")
    return normalized


def _normalize_custom_intent(intent: str) -> str:
    if not isinstance(intent, str):
        raise TypeError("custom intents must be strings")
    normalized = intent.strip().upper()
    if not _INTENT_PATTERN.fullmatch(normalized):
        raise ValueError("custom intents must match [A-Z][A-Z0-9_]{0,63}")
    return normalized


class CompanionEngine:
    """Run gate → retrieve → appraise → affect → generate with explicit seams.

    Gate and generation are model-driven and are not parity claims. The state
    transforms between them are deterministic and credential-free.
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        store: StateStore,
        conversation_id: str,
        retriever: MemoryRetriever | None = None,
        base_prompt: str = "",
        prompt_policy: PromptPolicy | None = None,
        turn_shape_policy: TurnShapePolicy | None = None,
        appraisal_policy: AppraisalPolicy = default_appraisal_policy,
        state_factory: Callable[[], CompanionState] = CompanionState,
        retrieval_limit: int = 6,
        max_retrieval_candidates: int = 64,
        gate_error_mode: GateErrorMode = "raise",
        custom_intents: Iterable[str] = (),
        limits: EngineLimits = DEFAULT_ENGINE_LIMITS,
    ) -> None:
        if not conversation_id:
            raise ValueError("conversation_id must not be empty")
        if isinstance(retrieval_limit, bool) or not isinstance(retrieval_limit, int):
            raise TypeError("retrieval_limit must be an integer")
        if retrieval_limit < 0:
            raise ValueError("retrieval_limit must be non-negative")
        if isinstance(max_retrieval_candidates, bool) or not isinstance(
            max_retrieval_candidates, int
        ):
            raise TypeError("max_retrieval_candidates must be an integer")
        if max_retrieval_candidates <= 0:
            raise ValueError("max_retrieval_candidates must be positive")
        if retrieval_limit > max_retrieval_candidates:
            raise ValueError("retrieval_limit must not exceed max_retrieval_candidates")
        if gate_error_mode not in ("raise", "respond", "silent"):
            raise ValueError("gate_error_mode must be 'raise', 'respond', or 'silent'")
        if not isinstance(limits, EngineLimits):
            raise TypeError("limits must be EngineLimits")
        self.model = model
        self.store = store
        self.retriever = retriever
        self.conversation_id = conversation_id
        self.base_prompt = base_prompt
        self.prompt_policy = prompt_policy or PromptPolicy()
        self.turn_shape_policy = turn_shape_policy or TurnShapePolicy()
        self.appraisal_policy = appraisal_policy
        self.state_factory = state_factory
        self.retrieval_limit = retrieval_limit
        self.max_retrieval_candidates = max_retrieval_candidates
        self.gate_error_mode = gate_error_mode
        self.custom_intents = frozenset(_normalize_custom_intent(item) for item in custom_intents)
        self.limits = limits

    async def turn(
        self,
        text: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Process and atomically persist one store-serialized conversation turn."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if len(text) > self.limits.max_message_chars:
            raise ValueError(f"message exceeds {self.limits.max_message_chars} characters")
        async with self.store.transaction(self.conversation_id) as transaction:
            history = await transaction.load_transcript()
            self._validate_history(history)
            state = await transaction.load_state() or self.state_factory()
            gate = await self._gate(text, history, state)
            user_message = Message("user", text)
            if not gate.should_respond:
                await transaction.commit(state=None, messages=(user_message,))
                return TurnResult(
                    text="",
                    intent=gate.intent,
                    silent=True,
                    mood=state.mood,
                )

            now = datetime.now(UTC)
            candidates: tuple[MemoryCandidate, ...] = ()
            if gate.should_retrieve and self.retriever is not None and self.retrieval_limit:
                retrieved = await self.retriever.retrieve(
                    RetrievalInput(
                        query=text,
                        history=history,
                        state=replace(state, carried_thought=None),
                        limit=self.retrieval_limit,
                        now=now,
                    )
                )
                candidates = self._validate_retrieval_batch(retrieved)
            memories = rank_candidates(
                candidates,
                limit=self.retrieval_limit,
                now=now,
                mood_valence=state.mood.valence,
            )

            appraisal = self.appraisal_policy(
                AppraisalPolicyInput(
                    state=state,
                    intent=gate.intent,
                    message=text,
                    expectation=state.expectation,
                )
            )
            if not isinstance(appraisal, AppraisalResult):
                raise TypeError("appraisal policies must return AppraisalResult")
            evolved = appraisal.state
            turn_shape = turn_shape_directive(
                evolved.mood,
                intent=gate.intent,
                history=history,
                emotions=appraisal.active_emotions,
                policy=self.turn_shape_policy,
            )
            system_prompt = build_system_prompt(
                self.base_prompt,
                evolved,
                PromptInputs(
                    emotions=appraisal.active_emotions,
                    turn_shape=turn_shape,
                ),
                policy=self.prompt_policy,
            )
            if len(system_prompt) > self.limits.max_prompt_chars:
                raise ValueError(f"system prompt exceeds {self.limits.max_prompt_chars} characters")
            request = GenerateInput(
                message=text,
                system_prompt=system_prompt,
                history=history,
                state=replace(evolved, carried_thought=None),
                intent=gate.intent,
                emotions=appraisal.active_emotions,
                decoding=decoding_params(evolved.mood),
                untrusted_context=build_untrusted_context(
                    evolved,
                    memories,
                    surface_carried_thought=not history,
                ),
            )
            chunks: list[str] = []
            output_chars = 0
            async for chunk in self.model.generate(request):
                if not isinstance(chunk, str):
                    raise TypeError("model adapters must yield strings")
                next_output_chars = output_chars + len(chunk)
                if next_output_chars > self.limits.max_output_chars:
                    raise ValueError(
                        f"model output exceeds {self.limits.max_output_chars} characters"
                    )
                chunks.append(chunk)
                output_chars = next_output_chars
                if on_token is not None:
                    on_token(chunk)
            response = "".join(chunks)
            await transaction.commit(
                state=evolved,
                messages=(user_message, Message("assistant", response)),
            )
            return TurnResult(
                text=response,
                intent=gate.intent,
                silent=False,
                mood=evolved.mood,
                emotions=appraisal.active_emotions,
                memories=memories,
            )

    def _validate_history(self, history: tuple[Message, ...]) -> None:
        if len(history) > self.limits.max_history_messages:
            raise ValueError(f"history exceeds {self.limits.max_history_messages} messages")
        history_chars = 0
        for message in history:
            history_chars += len(message.content)
            if history_chars > self.limits.max_history_chars:
                raise ValueError(f"history exceeds {self.limits.max_history_chars} characters")

    async def _gate(
        self,
        text: str,
        history: tuple[Message, ...],
        state: CompanionState,
    ) -> GateResult:
        try:
            raw = await self.model.gate(
                GateInput(
                    message=text,
                    history=history,
                    state=replace(state, carried_thought=None),
                )
            )
            if not isinstance(raw, GateResult):
                raise TypeError("model gate must return GateResult")
            return replace(
                raw,
                intent=normalize_intent(raw.intent, custom_intents=self.custom_intents),
            )
        except Exception:
            if self.gate_error_mode == "raise":
                raise
            return DEFAULT_GATE_RESULT if self.gate_error_mode == "respond" else SILENT_GATE_RESULT

    def _validate_retrieval_batch(
        self,
        candidates: Sequence[MemoryCandidate],
    ) -> tuple[MemoryCandidate, ...]:
        if not isinstance(candidates, Sequence):
            raise TypeError("memory retrievers must return a sequence")
        if len(candidates) > self.max_retrieval_candidates:
            raise ValueError(
                f"too many memory candidates: {len(candidates)} exceeds "
                f"{self.max_retrieval_candidates}"
            )
        if not all(isinstance(candidate, MemoryCandidate) for candidate in candidates):
            raise TypeError("memory retrievers must return only MemoryCandidate values")
        return tuple(candidates)

    async def presence(self, cognition: CognitionState | None = None) -> PresenceVector:
        """Load current state and return its deterministic presence vector."""
        async with self.store.transaction(self.conversation_id) as transaction:
            state = await transaction.load_state() or self.state_factory()
            return build_presence_vector(state, cognition)


__all__ = [
    "BUILTIN_INTENTS",
    "DEFAULT_ENGINE_LIMITS",
    "DEFAULT_GATE_RESULT",
    "SILENT_GATE_RESULT",
    "CompanionEngine",
    "EngineLimits",
    "GateErrorMode",
    "normalize_intent",
]
