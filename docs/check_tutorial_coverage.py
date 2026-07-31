#!/usr/bin/env python3
"""Validate quality metadata and core-package scope for Hyperion tutorials."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
INVENTORY_PATH = DOCS_DIR / "tutorial_inventory.json"
VALID_STATUSES = {"stable", "experimental"}
VALID_VALIDATION_TYPES = {
    "executable_fixture_test",
    "cli_smoke_test",
    "documented_prerequisite",
}
CORE_EGS_LINK = re.compile(
    r"(?::(?:doc|download):`[^`]*\begs/|\.\.\s+literalinclude::[^\n]*\begs/|"
    r"https?://[^\s>`]*\begs/)",
    re.IGNORECASE,
)


def expected_tutorial_paths() -> set[str]:
    """Return the maintained guided-documentation pages requiring inventory."""
    paths = {"getting-started.rst", "quickstart.rst", "np/speech_augmentation.rst"}
    paths.update(
        path.relative_to(DOCS_DIR).as_posix()
        for path in (DOCS_DIR / "how-to").glob("*.rst")
    )
    for path in DOCS_DIR.rglob("*.rst"):
        title = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
        if "tutorial" in title.lower() and path.name != "tutorial-quality.rst":
            paths.add(path.relative_to(DOCS_DIR).as_posix())
    return paths


def test_path_exists(value: object) -> bool:
    """Return whether a pytest node identifier points at an existing test file."""
    if not isinstance(value, str) or not value:
        return False
    return (REPO_DIR / value.split("::", maxsplit=1)[0]).is_file()


def main() -> int:
    """Validate tutorial classification, prerequisite visibility, and evidence."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    paths: set[str] = set()

    for tutorial in inventory.get("tutorials", []):
        path_value = tutorial.get("path", "<missing path>")
        if not isinstance(path_value, str) or not path_value:
            errors.append("tutorial record has no path")
            continue
        if path_value in paths:
            errors.append(f"{path_value}: duplicate tutorial record")
        paths.add(path_value)

        source_path = DOCS_DIR / path_value
        if source_path.suffix != ".rst" or not source_path.is_file():
            errors.append(f"{path_value}: tutorial page does not exist")
            continue
        source = source_path.read_text(encoding="utf-8")

        if tutorial.get("status") not in VALID_STATUSES:
            errors.append(f"{path_value}: invalid status {tutorial.get('status')!r}")
        if not tutorial.get("title"):
            errors.append(f"{path_value}: missing title")
        prerequisites = tutorial.get("prerequisites")
        if not isinstance(prerequisites, list) or not prerequisites:
            errors.append(f"{path_value}: missing prerequisites")
            prerequisites = []
        for prerequisite in prerequisites:
            if not isinstance(prerequisite, str) or prerequisite not in source:
                errors.append(f"{path_value}: prerequisite is not documented: {prerequisite!r}")
        outputs = tutorial.get("expected_outputs")
        if not isinstance(outputs, list) or not outputs:
            errors.append(f"{path_value}: missing expected_outputs")

        validation = tutorial.get("validation")
        if not isinstance(validation, dict):
            errors.append(f"{path_value}: missing validation metadata")
            continue
        validation_type = validation.get("type")
        if validation_type not in VALID_VALIDATION_TYPES:
            errors.append(f"{path_value}: invalid validation type {validation_type!r}")
        elif validation_type in {"executable_fixture_test", "cli_smoke_test"}:
            if not test_path_exists(validation.get("test")):
                errors.append(f"{path_value}: validation test does not exist: {validation.get('test')!r}")
        elif validation.get("requirement") not in prerequisites:
            errors.append(f"{path_value}: documented validation requirement is not a prerequisite")

        if tutorial.get("core_package") is not True:
            errors.append(f"{path_value}: tutorial must declare core_package explicitly")
        elif CORE_EGS_LINK.search(source):
            errors.append(f"{path_value}: core-package tutorial links to egs/")

    expected = expected_tutorial_paths()
    missing = sorted(expected - paths)
    unexpected = sorted(paths - expected)
    if missing:
        errors.append("unclassified tutorials: " + ", ".join(missing))
    if unexpected:
        errors.append("inventory pages outside the tutorial scope: " + ", ".join(unexpected))

    if errors:
        print("Tutorial coverage check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    print(f"Tutorial coverage check passed: {len(paths)} tutorials classified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
