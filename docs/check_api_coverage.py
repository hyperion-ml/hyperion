#!/usr/bin/env python3
"""Validate that every curated public API concept has documentation coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
INVENTORY_PATH = DOCS_DIR / "api_inventory.json"
VALID_STATUSES = {"stable", "experimental", "excluded"}


def main() -> int:
    """Validate the API inventory and report every missing coverage entry."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen_references: set[str] = set()

    for concept in inventory.get("concepts", []):
        name = concept.get("name", "<unnamed concept>")
        reference = concept.get("reference", "")
        status = concept.get("status")
        doc_paths = concept.get("docs", [])

        if not reference:
            errors.append(f"{name}: missing reference")
        elif reference in seen_references:
            errors.append(f"{name}: duplicate reference {reference!r}")
        seen_references.add(reference)

        if status not in VALID_STATUSES:
            errors.append(f"{name}: invalid status {status!r}")
        if not doc_paths:
            errors.append(f"{name}: no documentation pages assigned")
            continue

        referenced = False
        for relative_path in doc_paths:
            path = DOCS_DIR / relative_path
            if not path.is_file():
                errors.append(f"{name}: documentation page does not exist: {relative_path}")
                continue
            if reference in path.read_text(encoding="utf-8"):
                referenced = True

        if not referenced:
            errors.append(
                f"{name}: {reference!r} is not mentioned in its assigned documentation"
            )

    if errors:
        print("API coverage check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    print(f"API coverage check passed: {len(seen_references)} concepts classified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
