#!/usr/bin/env python3
"""Validate the required release-note categories and deprecation links."""

from __future__ import annotations

import re
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
RELEASE_NOTES_PATH = DOCS_DIR / "release-notes.rst"
INDEX_PATH = DOCS_DIR / "index.rst"
REQUIRED_HEADINGS = (
    "Unreleased",
    "Stable public API",
    "CLI commands",
    "Artifact and configuration compatibility",
    "Deprecations",
)
DEPRECATION_ENTRY = re.compile(
    r"\* \*\*Deprecated:\*\*.*?\*\*Replacement:\*\*\s+(:\w+:`[^`]+`)"
    r".*?\*\*Migration:\*\*\s+(:\w+:`[^`]+`)",
    re.DOTALL,
)


def main() -> int:
    """Check release-note structure and complete deprecation entries."""
    source = RELEASE_NOTES_PATH.read_text(encoding="utf-8")
    index = INDEX_PATH.read_text(encoding="utf-8")
    errors: list[str] = []

    if "   release-notes" not in index:
        errors.append("release-notes.rst is not in the documentation toctree")
    for heading in REQUIRED_HEADINGS:
        if heading not in source:
            errors.append(f"missing required release-notes heading: {heading}")

    for entry in re.finditer(r"^\* \*\*Deprecated:\*\*.*?(?=^\* |\Z)", source, re.MULTILINE | re.DOTALL):
        if not DEPRECATION_ENTRY.fullmatch(entry.group(0).strip()):
            errors.append(
                "deprecation entries require linked **Replacement:** and "
                "**Migration:** fields"
            )

    if errors:
        print("Release-notes check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    print("Release-notes check passed: required categories and deprecation links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
