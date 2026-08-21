# Security policy

## Supported versions

Security fixes are applied to the latest `0.1.x` release and to `main`.

| Version | Supported |
|---|---|
| `0.1.x` | Yes |
| `< 0.1` | No |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for the repository. Do not open a
public issue containing an exploit, credential, private conversation, or personal
data.

Include:

- affected runtime and version/commit
- minimal reproduction
- realistic impact
- whether any real data or credential was exposed

You should receive an acknowledgement within seven days. Coordinated disclosure
is preferred after a fix is available.

## Scope

The kernel has no built-in network client, authentication system, or durable
database. Security properties of model, storage, retrieval, and transport
adapters belong to the application integrating them. This project will still
accept reports about unsafe defaults, state-boundary violations, prompt injection
across adapter boundaries, data leakage in examples/tests, and dependency or
packaging vulnerabilities.
