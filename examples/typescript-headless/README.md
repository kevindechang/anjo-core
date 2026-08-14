# TypeScript headless example

From `typescript/`, run `npm run example`. The command packs the publishable
package, installs that tarball here under its public package name, then typechecks
and runs this three-turn consumer. It prints initial presence plus mood, ranked
evidence, and presence after each turn, using only credential-free in-memory
adapters and caller-owned synthetic wording.
