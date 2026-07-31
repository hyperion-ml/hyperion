#!/usr/bin/env python3
"""Validate landing and owner documentation for curated public namespaces."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
INVENTORY_PATH = DOCS_DIR / "namespace_inventory.json"
VALID_STATUSES = {"stable", "experimental", "excluded"}


def read_doc(relative_path: object, namespace: str, field: str, errors: list[str]) -> str:
    """Read an inventory documentation page and report invalid paths."""
    if not isinstance(relative_path, str) or not relative_path:
        errors.append(f"{namespace}: missing {field} page")
        return ""
    path = DOCS_DIR / relative_path
    if path.suffix != ".rst" or not path.is_file():
        errors.append(f"{namespace}: {field} page does not exist: {relative_path!r}")
        return ""
    return path.read_text(encoding="utf-8")


def main() -> int:
    """Check every curated namespace for classification and documentation."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    namespaces: set[str] = set()
    counts: Counter[str] = Counter()

    for item in inventory.get("namespaces", []):
        namespace = item.get("namespace", "<missing namespace>")
        if not isinstance(namespace, str) or not namespace:
            errors.append("namespace record has no namespace")
            continue
        if namespace in namespaces:
            errors.append(f"{namespace}: duplicate namespace record")
        namespaces.add(namespace)

        status = item.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{namespace}: invalid status {status!r}")
        else:
            counts[status] += 1

        if not item.get("description"):
            errors.append(f"{namespace}: missing description")
        if status == "excluded" and not item.get("exclusion_reason"):
            errors.append(f"{namespace}: excluded namespace needs exclusion_reason")

        landing_text = read_doc(item.get("landing"), namespace, "landing", errors)
        read_doc(item.get("owner"), namespace, "owner", errors)
        if landing_text and namespace not in landing_text:
            errors.append(
                f"{namespace}: landing page does not mention the namespace: "
                f"{item['landing']!r}"
            )

    if not namespaces:
        errors.append("namespace inventory is empty")
    for status in VALID_STATUSES:
        if counts[status] == 0:
            errors.append(f"namespace inventory has no {status} entries")

    if errors:
        print("Namespace coverage check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    print(
        "Namespace coverage check passed: "
        f"{len(namespaces)} namespaces classified "
        f"({counts['stable']} stable, {counts['experimental']} experimental, "
        f"{counts['excluded']} excluded)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
