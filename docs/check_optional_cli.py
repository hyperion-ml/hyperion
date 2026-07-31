#!/usr/bin/env python3
"""Check CLI help for commands that require optional dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from generate_cli_reference import capture_help, subcommands_from_help


DOCS_DIR = Path(__file__).resolve().parent
INVENTORY_PATH = DOCS_DIR / "cli_inventory.json"


def main() -> int:
    """Run ``--help`` for every CLI excluded from the required check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    commands = [
        item for item in inventory["commands"] if not item.get("cli_check", True)
    ]

    def capture(item: dict[str, object], arguments: list[str]) -> tuple[bool, str]:
        return capture_help(
            sys.executable,
            str(item["script"]),
            arguments,
            args.timeout,
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        top_level = list(executor.map(lambda item: capture(item, []), commands))

    failures: list[tuple[str, str]] = []
    subcommand_tasks: list[tuple[dict[str, object], str]] = []
    for item, (available, output) in zip(commands, top_level):
        if not available:
            failures.append((str(item["command"]), output))
        else:
            subcommand_tasks.extend(
                (item, subcommand)
                for subcommand in subcommands_from_help(output)
            )

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        subcommand_results = list(
            executor.map(
                lambda task: capture(task[0], [task[1]]), subcommand_tasks
            )
        )

    for (item, subcommand), (available, output) in zip(
        subcommand_tasks, subcommand_results
    ):
        if not available:
            failures.append((f"{item['command']} {subcommand}", output))

    if failures:
        print("Optional CLI check failed:", file=sys.stderr)
        for command, detail in failures:
            print(f"\n{command}:\n{detail}", file=sys.stderr)
        return 1

    print(f"Optional CLI check passed: {len(commands)} commands checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
