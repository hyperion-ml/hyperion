#!/usr/bin/env python3
"""Validate that maintained commands are visible in generated CLI documentation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
BIN_DIR = REPO_DIR / "hyperion" / "bin"
INVENTORY_PATH = DOCS_DIR / "cli_inventory.json"
TASK_INDEX_PATH = DOCS_DIR / "generated" / "cli-index.rst"
HELP_REFERENCE_PATH = DOCS_DIR / "generated" / "cli-reference.rst"
CLI_PAGE_PATH = DOCS_DIR / "cli.rst"
MAINTAINED_STATUSES = {"stable", "experimental"}


def main() -> int:
    """Check inventory coverage plus task-index and parser-reference visibility."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    task_index = TASK_INDEX_PATH.read_text(encoding="utf-8")
    help_reference = HELP_REFERENCE_PATH.read_text(encoding="utf-8")
    cli_page = CLI_PAGE_PATH.read_text(encoding="utf-8")
    source_scripts = {
        path.stem for path in BIN_DIR.glob("*.py") if path.stem != "__init__"
    }
    maintained = [
        command
        for command in inventory["commands"]
        if command["status"] in MAINTAINED_STATUSES
    ]
    classified_scripts = {command["script"] for command in inventory["commands"]}
    errors: list[str] = []

    missing_classification = sorted(source_scripts - classified_scripts)
    if missing_classification:
        errors.append(
            "hyperion/bin modules not classified: "
            + ", ".join(missing_classification)
        )

    if ".. include:: generated/cli-index.rst" not in cli_page:
        errors.append("cli.rst does not include the generated task index")

    for command in maintained:
        name = command["command"]
        script = command["script"]
        index_entry = re.compile(rf"^\* ``{re.escape(name)}``(?:\s|$)", re.MULTILINE)
        if not index_entry.search(task_index):
            errors.append(f"{name}: not visible in generated task index")

        help_section = re.compile(
            rf"^{re.escape(name)}\n-+\n\nModule: ``hyperion\.bin\.{re.escape(script)}``\.",
            re.MULTILINE,
        )
        if not help_section.search(help_reference):
            errors.append(f"{name}: missing generated parser/help section")

    if errors:
        print("CLI quality check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    print(
        "CLI quality check passed: "
        f"{len(maintained)} maintained commands are classified, indexed, and referenced."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
