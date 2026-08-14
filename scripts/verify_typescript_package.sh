#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TYPESCRIPT_DIR="$REPO_DIR/typescript"
EXAMPLE_DIR="$REPO_DIR/examples/typescript-headless"
ANJO_PACK_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$ANJO_PACK_DIR" "$EXAMPLE_DIR/node_modules" "$EXAMPLE_DIR/dist"
}
trap cleanup EXIT
cleanup
mkdir -p "$ANJO_PACK_DIR"

ANJO_TARBALL_NAME="$(npm pack --silent --pack-destination "$ANJO_PACK_DIR" "$TYPESCRIPT_DIR" | tail -n 1)"
ANJO_TARBALL="$ANJO_PACK_DIR/$ANJO_TARBALL_NAME"
test -f "$ANJO_TARBALL"

npm install \
  --prefix "$EXAMPLE_DIR" \
  --ignore-scripts \
  --package-lock=false \
  --no-save \
  "$ANJO_TARBALL"
"$TYPESCRIPT_DIR/node_modules/.bin/tsc" -p "$EXAMPLE_DIR/tsconfig.json"
node "$EXAMPLE_DIR/dist/main.js"

if tar -tzf "$ANJO_TARBALL" | grep -Eq '(\.map$|/src/)'; then
  echo "Packed TypeScript artifact exposes maps or source files" >&2
  exit 1
fi

(
  cd "$EXAMPLE_DIR"
  node --input-type=module -e '
    try {
      await import("@anjo-ai/core/internal/round");
      process.exitCode = 1;
    } catch (error) {
      if (error?.code !== "ERR_PACKAGE_PATH_NOT_EXPORTED") throw error;
    }
  '
)

echo "TypeScript package and clean consumer verified."
