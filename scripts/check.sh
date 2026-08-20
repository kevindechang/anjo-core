#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
fi

"$PYTHON_BIN" "$REPO_DIR/scripts/verify_public_boundary.py" "$REPO_DIR"
"$PYTHON_BIN" -m pytest \
  "$REPO_DIR/python/tests" \
  "$REPO_DIR/scripts/tests" \
  --cov=affect_kernel \
  --cov-branch \
  --cov-report=term-missing
"$PYTHON_BIN" -m ruff check \
  "$REPO_DIR/python" \
  "$REPO_DIR/scripts" \
  "$REPO_DIR/examples" \
  "$REPO_DIR/bench"
"$PYTHON_BIN" -m ruff format --check \
  "$REPO_DIR/python" \
  "$REPO_DIR/scripts" \
  "$REPO_DIR/examples" \
  "$REPO_DIR/bench"
"$PYTHON_BIN" -m mypy \
  --config-file "$REPO_DIR/python/pyproject.toml" \
  "$REPO_DIR/python/src"
# The examples are executable documentation; each asserts its own invariants.
"$PYTHON_BIN" "$REPO_DIR/examples/python-headless/main.py" > /dev/null
"$PYTHON_BIN" "$REPO_DIR/examples/game-npc/main.py" > /dev/null
# The benchmark report is generated. Fail if the checked-in numbers have drifted
# from what the code produces, so documentation can never quote stale results.
"$PYTHON_BIN" "$REPO_DIR/bench/run.py" --check
npm test --prefix "$REPO_DIR/typescript"
npm run typecheck --prefix "$REPO_DIR/typescript"
npm run test:coverage --prefix "$REPO_DIR/typescript"
PYTHON_BIN="$PYTHON_BIN" "$REPO_DIR/scripts/verify_python_package.sh"
"$REPO_DIR/scripts/verify_typescript_package.sh"
