.PHONY: setup test lint package check examples

setup:
	./scripts/setup.sh

test:
	python -m pytest python/tests scripts/tests
	npm test --prefix typescript

lint:
	python -m ruff check python scripts examples/python-headless
	python -m ruff format --check python scripts examples/python-headless
	python -m mypy --config-file python/pyproject.toml python/src
	npm run typecheck --prefix typescript

package:
	PYTHON_BIN=python ./scripts/verify_python_package.sh
	./scripts/verify_typescript_package.sh

check:
	./scripts/check.sh

examples:
	python examples/python-headless/main.py
	npm run example --prefix typescript
