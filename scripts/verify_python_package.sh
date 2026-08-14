#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ANJO_PACKAGE_DIR="$(mktemp -d)"
ANJO_INSTALL_DIR="$(mktemp -d)"
ANJO_SDIST_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$ANJO_PACKAGE_DIR" "$ANJO_INSTALL_DIR" "$ANJO_SDIST_DIR"
}
trap cleanup EXIT

"$PYTHON_BIN" -m build --no-isolation "$REPO_DIR/python" --outdir "$ANJO_PACKAGE_DIR"
"$PYTHON_BIN" -m twine check "$ANJO_PACKAGE_DIR"/*
"$PYTHON_BIN" -m check_wheel_contents "$ANJO_PACKAGE_DIR"/*.whl

"$PYTHON_BIN" -m venv "$ANJO_INSTALL_DIR/venv"
"$ANJO_INSTALL_DIR/venv/bin/python" -m pip install --no-deps "$ANJO_PACKAGE_DIR"/*.whl
"$ANJO_INSTALL_DIR/venv/bin/python" -c \
  'import anjo_core; assert anjo_core.__version__ == "0.1.0"'
"$ANJO_INSTALL_DIR/venv/bin/python" "$REPO_DIR/examples/python-headless/main.py"
"$ANJO_INSTALL_DIR/venv/bin/python" "$REPO_DIR/examples/game-npc/main.py"

tar -xzf "$ANJO_PACKAGE_DIR"/*.tar.gz -C "$ANJO_SDIST_DIR"
ANJO_SDIST_ROOT="$(find "$ANJO_SDIST_DIR" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [ -z "$ANJO_SDIST_ROOT" ]; then
  echo "Could not find the extracted source distribution" >&2
  exit 1
fi
PYTHONPATH="$ANJO_SDIST_ROOT/src" \
  "$PYTHON_BIN" -m pytest "$ANJO_SDIST_ROOT/tests"

echo "Python wheel and source distribution verified."
