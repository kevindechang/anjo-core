#!/usr/bin/env python3
"""Fail closed when private/product material enters the public core tree."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from build_kernel_fixture import validate_public_fixture
    from validate_continuity_fixture import validate_continuity_fixture
except ModuleNotFoundError:  # Imported as scripts.verify_public_boundary in tests.
    from scripts.build_kernel_fixture import validate_public_fixture
    from scripts.validate_continuity_fixture import validate_continuity_fixture

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitleaks.toml",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "ROADMAP.md",
    "SECURITY.md",
    "bench",
    "docs",
    "examples",
    "python",
    "scripts",
    "shared",
    "typescript",
}
IGNORED_PARTS = {
    ".DS_Store",  # Untracked macOS cruft; still an error below if it is tracked.
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".test-dist",
    "build",
    "coverage",
    "dist",
    "htmlcov",
}
IGNORED_PREFIXES = (".coverage",)
FORBIDDEN_PARTS = {
    "data",
    "deliverables",
    "marketing",
    "mobile",
    "ops",
    ".claude",
    ".codex",
    ".agents",
    ".shannon",
}
FORBIDDEN_SUFFIXES = {".pem", ".p12", ".key", ".sqlite", ".sqlite3", ".db"}
ALLOWED_TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".typed",
    ".yaml",
    ".yml",
}
ALLOWED_TEXT_NAMES = {".gitignore", "LICENSE", "Makefile"}
SECRET_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI API key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned sensitive env": re.compile(
        r"(?im)^[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)"
        r"\s*=\s*(?!example|placeholder|replace|your[-_])[^\s<][^\s]{11,}"
    ),
}
IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SSH_TARGET_PATTERN = re.compile(
    r"(?im)^\s*ssh\b[^\n]*\s+[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+(?:\s|$)"
)
DEPLOY_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|Users|srv|var/www)/[^\s`'\"]+"
)


def _is_ignored(rel: Path) -> bool:
    return any(
        part in IGNORED_PARTS or part.startswith(IGNORED_PREFIXES) for part in rel.parts
    )


def _tracked_files(root: Path) -> tuple[set[Path], str | None]:
    if not (root / ".git").exists():
        return set(), None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        return set(), f"could not inspect tracked files: {exc}"
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        return (
            set(),
            f"could not inspect tracked files: {detail or 'git ls-files failed'}",
        )
    paths = {Path(os.fsdecode(raw)) for raw in result.stdout.split(b"\0") if raw}
    return paths, None


def _iter_files(root: Path, tracked: set[Path]):
    seen: set[Path] = set()
    for rel in sorted(tracked, key=lambda path: path.as_posix()):
        path = root / rel
        seen.add(rel)
        if path.is_symlink():
            yield path, "symlink", True
        elif path.is_file():
            yield path, "file", True
        else:
            yield path, "missing", True

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel in seen or _is_ignored(rel):
            continue
        if path.is_symlink():
            yield path, "symlink", False
        elif path.is_file():
            yield path, "file", False


def verify(root: Path) -> list[str]:
    errors: list[str] = []
    tracked, git_error = _tracked_files(root)
    if git_error:
        errors.append(git_error)
    actual_top = {
        path.name
        for path in root.iterdir()
        if path.name != ".git"
        and path.name not in IGNORED_PARTS
        and not path.name.startswith(IGNORED_PREFIXES)
    }
    unexpected = sorted(actual_top - ALLOWED_TOP_LEVEL)
    if unexpected:
        errors.append(f"unexpected top-level entries: {', '.join(unexpected)}")

    for path, kind, is_tracked in _iter_files(root, tracked):
        rel = path.relative_to(root)
        if kind == "missing":
            errors.append(f"tracked path is missing: {rel}")
            continue
        if kind == "symlink":
            errors.append(f"symlink is not allowed: {rel}")
            continue
        if is_tracked and _is_ignored(rel):
            errors.append(f"generated/ignored path must not be tracked: {rel}")
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            errors.append(f"private/product path is not allowed: {rel}")
        if path.name.startswith(".env") or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"credential/data file is not allowed: {rel}")
        if (
            path.name not in ALLOWED_TEXT_NAMES
            and path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES
        ):
            errors.append(f"non-source file type is not allowed: {rel}")
        if path.stat().st_size > 2_000_000:
            errors.append(f"unexpected large file: {rel}")
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw:
            errors.append(f"NUL-containing/binary file is not allowed: {rel}")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 file is not allowed: {rel}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} pattern in {rel}")
        for match in IPV4_PATTERN.finditer(text):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if address.is_global:
                errors.append(f"public IPv4 literal in {rel}")
                break
        if SSH_TARGET_PATTERN.search(text):
            errors.append(f"SSH deployment target in {rel}")
        if DEPLOY_PATH_PATTERN.search(text):
            errors.append(f"absolute deployment path in {rel}")

    fixture_path = root / "shared/golden/kernel_golden.json"
    if not fixture_path.exists():
        errors.append("missing shared/golden/kernel_golden.json")
    else:
        try:
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            validate_public_fixture(fixture)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid public fixture: {exc}")

    continuity_path = root / "shared/golden/continuity_traces.json"
    if not continuity_path.exists():
        errors.append("missing shared/golden/continuity_traces.json")
    else:
        try:
            continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
            validate_continuity_fixture(continuity)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid continuity fixture: {exc}")
    return errors


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    errors = verify(root)
    if errors:
        print("PUBLIC BOUNDARY: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PUBLIC BOUNDARY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
