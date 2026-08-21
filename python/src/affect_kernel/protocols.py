"""Adapter protocols that keep I/O and model inference outside the kernel."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from .appraisal import AppraisalPolicyInput, AppraisalResult
from .models import (
    AffectState,
    GateInput,
    GateResult,
    GenerateInput,
    MemoryCandidate,
    Message,
    RetrievalInput,
)


@runtime_checkable
class AppraisalPolicy(Protocol):
    """Synchronous domain-event to deterministic appraisal seam."""

    def __call__(self, request: AppraisalPolicyInput, /) -> AppraisalResult: ...


@runtime_checkable
class ModelAdapter(Protocol):
    async def gate(self, request: GateInput) -> GateResult:
        """Classify intent and decide whether to respond or retrieve."""
        ...

    def generate(self, request: GenerateInput) -> AsyncIterator[str]:
        """Yield chunks, keeping untrusted_context in a user/tool-data channel.

        Adapters must not concatenate untrusted context, state, or retrieval
        evidence into ``system_prompt``. Empty output is allowed and visible.
        """
        ...


@runtime_checkable
class ConversationTransaction(Protocol):
    """One serialized conversation snapshot with an atomic commit boundary."""

    async def load_state(self) -> AffectState | None: ...

    async def load_transcript(self) -> tuple[Message, ...]: ...

    async def commit(
        self,
        *,
        state: AffectState | None,
        messages: Sequence[Message],
    ) -> None:
        """Atomically persist the optional state update and all messages."""
        ...


@runtime_checkable
class StateStore(Protocol):
    def transaction(
        self, conversation_id: str
    ) -> AbstractAsyncContextManager[ConversationTransaction]:
        """Serialize all turns for this conversation, including across engine instances."""
        ...

    async def load_state(self, conversation_id: str) -> AffectState | None: ...

    async def save_state(self, conversation_id: str, state: AffectState) -> None: ...

    async def load_transcript(self, conversation_id: str) -> tuple[Message, ...]: ...

    async def append_message(self, conversation_id: str, message: Message) -> None: ...


@runtime_checkable
class MemoryRetriever(Protocol):
    async def retrieve(self, request: RetrievalInput) -> Sequence[MemoryCandidate]:
        """Return candidates; the kernel, not the adapter, owns their final score."""
        ...
