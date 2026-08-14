"""Credential-free headless conversation through the complete injected engine."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

from anjo_core import (
    CompanionEngine,
    CompanionState,
    GateResult,
    MemoryCandidate,
    PADMood,
)
from anjo_core.adapters import (
    InMemoryStateStore,
    ScriptedModelAdapter,
    StaticMemoryRetriever,
)


async def main() -> None:
    conversation_id = "headless-demo"
    store = InMemoryStateStore(
        states={conversation_id: CompanionState(mood=PADMood(0.1, 0.05, 0.0))}
    )
    model = ScriptedModelAdapter(
        gates=[
            GateResult(intent="CURIOSITY", should_respond=True, should_retrieve=True),
            GateResult(
                intent="VULNERABILITY", should_respond=True, should_retrieve=False
            ),
            GateResult(intent="CURIOSITY", should_respond=True, should_retrieve=False),
        ],
        responses=[
            (
                "That thread still has some pull. ",
                "A fresh attempt can change its shape.",
            ),
            ("A rough page can still be evidence that you returned.",),
            ("Wanting to try tomorrow is the part worth carrying forward.",),
        ],
    )
    retriever = StaticMemoryRetriever(
        [
            MemoryCandidate(
                id="demo-memory",
                text="The user has been learning to sketch.",
                distance=0.18,
                timestamp=(datetime.now(UTC) - timedelta(days=2)).isoformat(),
                significance=0.6,
            )
        ]
    )
    engine = CompanionEngine(
        model=model,
        store=store,
        retriever=retriever,
        conversation_id=conversation_id,
        base_prompt="You are a thoughtful synthetic companion in a local demonstration.",
    )

    turns = (
        "I picked up my sketchbook again.",
        "The first page looks terrible.",
        "Still, I want to try again tomorrow.",
    )

    print("Initial presence:")
    print(json.dumps((await engine.presence()).to_dict(), indent=2))
    for index, message in enumerate(turns, start=1):
        print(f"\nTurn {index} — User: {message}")
        print("Companion: ", end="", flush=True)
        result = await engine.turn(
            message,
            on_token=lambda chunk: print(chunk, end="", flush=True),
        )
        print()
        evidence = [memory.candidate.id for memory in result.memories]
        mood = result.mood
        print(
            f"State: intent={result.intent}, "
            f"PAD=({mood.valence:.4f}, {mood.arousal:.4f}, {mood.dominance:.4f}), "
            f"evidence={evidence or 'none'}"
        )
        presence = await engine.presence()
        print(f"Presence: {presence.line} ({presence.mode})")


if __name__ == "__main__":
    asyncio.run(main())
