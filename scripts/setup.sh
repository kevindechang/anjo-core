#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"Affect Kernel requires Python 3.11+; {sys.executable} is "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
'
node -e '
const major = Number(process.versions.node.split(".")[0]);
if (major < 20) {
  console.error(`Affect Kernel requires Node.js 20+; found ${process.versions.node}`);
  process.exit(1);
}
'

"$PYTHON_BIN" -m venv "$REPO_DIR/.venv"
"$REPO_DIR/.venv/bin/python" -m pip install \
  --require-hashes \
  -r "$REPO_DIR/python/requirements-dev.lock"
"$REPO_DIR/.venv/bin/python" -m pip install \
  --no-deps \
  --no-build-isolation \
  -e "$REPO_DIR/python"
npm ci --ignore-scripts --prefix "$REPO_DIR/typescript"

echo "Setup complete. Run: $REPO_DIR/scripts/check.sh"
