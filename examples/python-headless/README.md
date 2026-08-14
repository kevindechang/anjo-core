# Python headless example

This example uses the package's in-memory store, static retriever, and scripted
model adapter. It prints initial presence and the mood, ranked evidence, and
presence after each of three turns. It makes no network calls and needs no
credentials.

From the repository root:

```bash
python -m pip install -e ./python
python examples/python-headless/main.py
```

Replace the adapters—or inject an `AppraisalPolicy`—to embed the same
deterministic pipeline in another conversational domain.
Production model adapters must send `GenerateInput.untrusted_context` as
user/tool evidence rather than system instructions. Production stores must make
their per-conversation `transaction()` serialize turns and commit state plus the
message batch atomically.
