"""Credential-free in-memory adapters for tests, examples, and prototyping."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy

from ..models import AffectState, MemoryCandidate, Message, RetrievalInput


class _InMemoryTransaction:
    def __init__(self, store: InMemoryStateStore, conversation_id: str) -> None:
        self._store = store
        self._conversation_id = conversation_id
        self._committed = False

    async def load_state(self) -> AffectState | None:
        return deepcopy(self._store._states.get(self._conversation_id))

    async def load_transcript(self) -> tuple[Message, ...]:
        return tuple(deepcopy(self._store._transcripts.get(self._conversation_id, ())))

    async def commit(
        self,
        *,
        state: AffectState | None,
        messages: Sequence[Message],
    ) -> None:
        if self._committed:
            raise RuntimeError("a conversation transaction can only commit once")
        if state is not None and not isinstance(state, AffectState):
            raise TypeError("state must be AffectState or None")
        copied_messages = list(deepcopy(tuple(messages)))
        if not all(isinstance(message, Message) for message in copied_messages):
            raise TypeError("messages must contain only Message values")
        next_transcript = list(self._store._transcripts.get(self._conversation_id, ()))
        next_transcript.extend(copied_messages)
        copied_state = deepcopy(state) if state is not None else None

        # No await occurs between assignments: observers under the same store lock
        # can only see the complete pre-commit or post-commit snapshot.
        if copied_state is not None:
            self._store._states[self._conversation_id] = copied_state
        self._store._transcripts[self._conversation_id] = next_transcript
        self._committed = True


class InMemoryStateStore:
    """A process-local store with defensive copies at every protocol boundary."""

    def __init__(
        self,
        *,
        states: Mapping[str, AffectState] | None = None,
        transcripts: Mapping[str, Sequence[Message]] | None = None,
    ) -> None:
        self._states = deepcopy(dict(states or {}))
        self._transcripts = {
            key: list(deepcopy(messages)) for key, messages in (transcripts or {}).items()
        }
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conversation_id, asyncio.Lock())

    @asynccontextmanager
    async def transaction(self, conversation_id: str) -> AsyncIterator[_InMemoryTransaction]:
        """Hold the store-owned per-conversation lock for one whole turn."""
        async with self._lock_for(conversation_id):
            yield _InMemoryTransaction(self, conversation_id)

    async def load_state(self, conversation_id: str) -> AffectState | None:
        async with self._lock_for(conversation_id):
            return deepcopy(self._states.get(conversation_id))

    async def save_state(self, conversation_id: str, state: AffectState) -> None:
        async with self._lock_for(conversation_id):
            self._states[conversation_id] = deepcopy(state)

    async def load_transcript(self, conversation_id: str) -> tuple[Message, ...]:
        async with self._lock_for(conversation_id):
            return tuple(deepcopy(self._transcripts.get(conversation_id, ())))

    async def append_message(self, conversation_id: str, message: Message) -> None:
        async with self._lock_for(conversation_id):
            self._transcripts.setdefault(conversation_id, []).append(deepcopy(message))


class StaticMemoryRetriever:
    """Return a fixed candidate slice; useful when demonstrating deterministic ranking."""

    def __init__(self, candidates: Sequence[MemoryCandidate]) -> None:
        self.candidates = tuple(deepcopy(candidates))
        self.requests: list[RetrievalInput] = []

    async def retrieve(self, request: RetrievalInput) -> tuple[MemoryCandidate, ...]:
        self.requests.append(request)
        return tuple(deepcopy(self.candidates))


__all__ = ["InMemoryStateStore", "StaticMemoryRetriever"]
