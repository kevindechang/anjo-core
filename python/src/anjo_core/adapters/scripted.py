"""A deterministic, credential-free model adapter for examples and tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from ..models import GateInput, GateResult, GenerateInput


class ScriptedModelAdapter:
    """Consume queued gate results and queued response chunks in call order."""

    def __init__(
        self,
        *,
        gates: Sequence[GateResult] = (),
        responses: Sequence[Sequence[str] | str] = (),
        gate_errors: Sequence[Exception] = (),
        default_gate: GateResult | None = None,
    ) -> None:
        self._gates = list(gates)
        self._responses = [
            (response,) if isinstance(response, str) else tuple(response) for response in responses
        ]
        self._gate_errors = list(gate_errors)
        self.default_gate = default_gate or GateResult()
        self.gate_requests: list[GateInput] = []
        self.requests: list[GenerateInput] = []

    async def gate(self, request: GateInput) -> GateResult:
        self.gate_requests.append(request)
        if self._gate_errors:
            raise self._gate_errors.pop(0)
        if self._gates:
            return self._gates.pop(0)
        return self.default_gate

    async def generate(self, request: GenerateInput) -> AsyncIterator[str]:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("no scripted response remains")
        for chunk in self._responses.pop(0):
            await asyncio.sleep(0)
            yield chunk


__all__ = ["ScriptedModelAdapter"]
