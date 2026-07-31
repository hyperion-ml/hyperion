#!/usr/bin/env python3
"""Validate the maintained CLI inventory against the command source tree."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
BIN_DIR = REPO_DIR / "hyperion" / "bin"
INVENTORY_PATH = DOCS_DIR / "cli_inventory.json"
PYPROJECT_PATH = REPO_DIR / "pyproject.toml"
VALID_STATUSES = {"stable", "experimental", "excluded"}


def main() -> int:
    """Check classification, documentation assignment, and installed command names."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    entry_points = project["project"]["scripts"]
    source_scripts = {path.stem for path in BIN_DIR.glob("*.py") if path.stem != "__init__"}
    errors: list[str] = []
    seen_scripts: set[str] = set()
    seen_commands: set[str] = set()

    for item in inventory.get("commands", []):
        script = item.get("script", "<missing script>")
        command = item.get("command", "<missing command>")
        status = item.get("status")
        guide = item.get("guide", "")

        if script in seen_scripts:
            errors.append(f"{script}: duplicate inventory script")
        seen_scripts.add(script)
        if command in seen_commands:
            errors.append(f"{command}: duplicate installed command")
        seen_commands.add(command)
        if status not in VALID_STATUSES:
            errors.append(f"{script}: invalid status {status!r}")

        # Keep this rule aligned with generate_pyproject.py: utility modules
        # named hyperion_dataset.py and hyperion_tables.py intentionally become
        # hyperion-dataset and hyperion-tables, rather than a doubled prefix.
        expected_command = "hyperion-" + script.removeprefix("hyperion_").replace(
            "_", "-"
        )
        if command != expected_command:
            errors.append(f"{script}: command is {command!r}, expected {expected_command!r}")
        if script not in source_scripts:
            errors.append(f"{script}: no matching hyperion/bin module")

        if status == "excluded":
            if not item.get("exclusion_reason"):
                errors.append(f"{script}: excluded commands need exclusion_reason")
            continue

        if not guide:
            errors.append(f"{script}: documented command has no guide")
        elif not (DOCS_DIR / guide).is_file():
            errors.append(f"{script}: guide does not exist: {guide}")
        entry_point = entry_points.get(command)
        expected_entry_point = f"hyperion.bin.{script}:main"
        if entry_point != expected_entry_point:
            errors.append(
                f"{script}: pyproject entry point is {entry_point!r}, "
                f"expected {expected_entry_point!r}"
            )

    missing = sorted(source_scripts - seen_scripts)
    unexpected = sorted(seen_scripts - source_scripts)
    if missing:
        errors.append("unclassified hyperion/bin modules: " + ", ".join(missing))
    if unexpected:
        errors.append("inventory modules absent from hyperion/bin: " + ", ".join(unexpected))

    if errors:
        print("CLI coverage check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    print(f"CLI coverage check passed: {len(seen_scripts)} command modules classified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
