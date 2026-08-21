# Governance

Affect Kernel currently uses a maintainer-led model.

- Maintainers set scope, merge changes, cut releases, and resolve security issues.
- Significant public API or parity-contract changes should begin as an issue or
  short design proposal.
- Decisions favor a small deterministic kernel, explicit adapter boundaries,
  behavioral evidence, and cross-runtime clarity.
- Provider integrations, application policy, and branded personas stay outside
  the kernel unless they establish a broadly reusable interface.

As sustained contributors emerge, maintainership can be granted based on review
quality, reliability, and care for the public boundary. Governance changes are
made through pull requests to this file.

## Bandwidth

Worth knowing before you invest time in a contribution: this is maintained by
one person, alongside other work. Expect a first response to an issue or pull
request within about a week, and longer for anything touching the parity
contract, which needs a reviewed fixture change in both runtimes.

Things that get looked at fastest, in order:

1. A reproducible bug in a deterministic transform, with a failing case.
2. A benchmark or evaluation result — including one that contradicts something
   this repository claims. `docs/foundations.md` lists the results that would
   falsify its own choices, and the linear-recency claim has already been
   retracted on evidence once.
3. A new storage or model adapter, or a domain preset.
4. API surface changes, which need a design discussion first.

If something here is stalled and you need it, say so on the issue rather than
assuming it was rejected.

