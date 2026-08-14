# Provenance and release boundary

Anjo Core is a clean extraction of reusable deterministic behavior developed for
the Anjo application. The public repository starts with new history; it does not
inherit the application repository's commits, deployment material, product
configuration, prompts, or user data.

## Included material

- Independently packaged Python and TypeScript implementations of the public
  affect, appraisal, retrieval-scoring, surfacing, and orchestration contracts.
- A sanitized 225-case component corpus derived from deterministic application
  behavior. Product prose and application-only policy are excluded.
- Three wholly synthetic longitudinal traces covering engagement,
  conflict/recovery, and affect decay.
- Original examples and documentation written for this standalone repository.

No dependency source or model output is vendored. Build and test dependencies
are declared in the package manifests and lockfiles.

## Initial-publication checklist

Before making a remote repository public, the releasing maintainer must:

1. Confirm that the project has the right to publish and license every extracted
   implementation and fixture.
2. Confirm the intended license before accepting outside contributions. The
   current manifests consistently declare Apache-2.0.
3. Run `./scripts/check.sh` from a clean checkout.
4. Inspect `git ls-files`, the Python wheel and source distribution, and the npm
   tarball rather than relying on ignore rules.
5. Run the pinned Gitleaks policy over the complete new Git history.
6. Publish from this clean repository only; never copy or rewrite the private
   application's Git history into it.

The technical checks can establish a clean boundary. The ownership and license
confirmation remains a maintainer responsibility and cannot be inferred from a
secret scan.
