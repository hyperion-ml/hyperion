#!/usr/bin/env python3
"""Generate the CLI option reference from installed command help output."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
INVENTORY_PATH = DOCS_DIR / "cli_inventory.json"
OUTPUT_PATH = DOCS_DIR / "generated" / "cli-reference.rst"


def rst_heading(text: str, adornment: str) -> list[str]:
    """Return an RST heading."""
    return [text, adornment * len(text), ""]


def capture_help(
    python: str, script: str, arguments: list[str], timeout: int
) -> tuple[bool, str]:
    """Capture help for one command invocation."""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_DIR / "docs" / "_build" / ".matplotlib"))
    # argparse uses the terminal width when wrapping help text. The subprocess
    # has no TTY, so fix the fallback width to make generated output portable.
    env["COLUMNS"] = "80"
    env["LINES"] = "24"
    # Some CLI imports load librosa/numba. JIT compilation is unnecessary for
    # help capture and can fail while importing packages from non-standard
    # filesystem paths.
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        completed = subprocess.run(
            [python, "-m", f"hyperion.bin.{script}", *arguments, "--help"],
            cwd=REPO_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout} seconds while loading parser help."

    # jsonargparse uses a backspace control character in the help notation for
    # optional values. It is meaningful to terminals but not to Sphinx source.
    output = normalise_output(completed.stdout).strip()
    if completed.returncode == 0 and output:
        return True, output

    detail = normalise_output(completed.stderr).strip() or output or f"exit status {completed.returncode}"
    return False, detail


def normalise_output(output: str) -> str:
    """Remove host-specific terminal output before committing a help snapshot."""
    output = output.replace("\x08", "")
    output = output.replace(str(REPO_DIR), "<repository>")
    output = output.replace(str(Path.home()), "<home>")
    return re.sub(r"0x[0-9A-Fa-f]+", "<address>", output)


def subcommands_from_help(help_text: str) -> list[str]:
    """Extract jsonargparse's listed subcommands from top-level help text."""
    marker = "Available subcommands:"
    if marker not in help_text:
        return []

    entries: list[str] = []
    for line in help_text.split(marker, maxsplit=1)[1].splitlines()[1:]:
        match = re.match(r"^\s{4}([A-Za-z0-9][A-Za-z0-9_-]*)\s*$", line)
        if match:
            entries.append(match.group(1))
        elif entries and line.strip():
            break
    return entries


def add_help_block(lines: list[str], output: str) -> None:
    """Append captured terminal help as a literal RST block."""
    lines.extend([".. code-block:: text", ""])
    lines.extend(f"   {line}" if line else "" for line in output.splitlines())
    lines.append("")


def normalize_for_comparison(content: str) -> str:
    """Ignore terminal wrapping and spacing differences in help blocks."""

    # Python's repr for locally-created lambdas varies between contexts and
    # Python versions (``<lambda>`` vs ``<locals>.<lambda>``). It carries no
    # useful CLI information, so remove that unstable qualifier.
    content = content.replace("<locals>.", "")
    normalized: list[str] = []
    help_lines: list[str] = []

    def flush_help() -> None:
        if help_lines:
            # Help output is terminal-formatted.  Different terminal widths
            # can move a word to another line or alter indentation, without
            # changing the CLI itself.  Compare the block as tokens while
            # keeping the surrounding RST structure significant.
            normalized.append(" ".join(" ".join(help_lines).split()))
            help_lines.clear()

    in_help = False
    for line in content.splitlines():
        if line == ".. code-block:: text":
            flush_help()
            in_help = True
            normalized.append(line)
        elif in_help and (line.startswith("   ") or not line):
            if line:
                help_lines.append(line[3:])
        else:
            flush_help()
            in_help = False
            normalized.append(line.rstrip())
    flush_help()
    return "\n".join(normalized)


def render(
    python: str,
    timeout: int,
    allow_unavailable: bool,
    selected_scripts: set[str] | None,
    jobs: int,
    include_optional: bool,
) -> tuple[str, list[str]]:
    """Render every inventory command and return unavailable command names."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    unavailable: list[str] = []
    lines = [
        ".. This file is generated by docs/generate_cli_reference.py; do not edit it directly.",
        "",
        *rst_heading("Generated Command and Option Reference", "="),
        "Each section is captured from the installed command's ``--help`` output.",
        "For commands with jsonargparse subcommands, the reference also captures each",
        "listed subcommand's ``--help`` output.",
        "",
    ]

    commands = sorted(inventory["commands"], key=lambda item: item["command"])
    if not include_optional:
        commands = [command for command in commands if command.get("cli_check", True)]
    if selected_scripts:
        commands = [item for item in commands if item["script"] in selected_scripts]
    def capture(command: dict[str, object], arguments: list[str]) -> tuple[bool, str]:
        return capture_help(python, str(command["script"]), arguments, timeout)

    print(f"Capturing top-level help for {len(commands)} command(s) with {jobs} worker(s).")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        top_level_results = list(
            executor.map(lambda command: capture(command, []), commands)
        )

    subcommand_tasks: list[tuple[dict[str, object], str]] = []
    for command, (available, output) in zip(commands, top_level_results):
        if available:
            subcommand_tasks.extend(
                (command, subcommand) for subcommand in subcommands_from_help(output)
            )
    print(f"Capturing help for {len(subcommand_tasks)} discovered subcommand(s).")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        subcommand_results = list(
            executor.map(
                lambda task: capture(task[0], [task[1]]), subcommand_tasks
            )
        )
    subcommand_result_map = {
        (str(command["script"]), subcommand): result
        for (command, subcommand), result in zip(subcommand_tasks, subcommand_results)
    }

    for index, (command, (available, output)) in enumerate(
        zip(commands, top_level_results), start=1
    ):
        print(f"[{index}/{len(commands)}] {command['command']}", flush=True)
        lines.extend(rst_heading(command["command"], "-"))
        lines.append(f"Module: ``hyperion.bin.{command['script']}``.")
        lines.append(f"Support level: **{command['status']}**.")
        if command["optional_requirements"]:
            lines.append(
                "Conditional runtime requirements: "
                + ", ".join(f"``{item}``" for item in command["optional_requirements"])
                + "."
            )
        lines.append("")

        if available:
            add_help_block(lines, output)
            for subcommand in subcommands_from_help(output):
                lines.extend(rst_heading(f"{command['command']} {subcommand}", "~"))
                subcommand_available, subcommand_output = subcommand_result_map[
                    (str(command["script"]), subcommand)
                ]
                if subcommand_available:
                    add_help_block(lines, subcommand_output)
                    continue
                unavailable.append(f"{command['command']} {subcommand}")
                lines.extend(
                    [
                        "Parser help was unavailable in the generating environment:",
                        "",
                    ]
                )
                add_help_block(lines, subcommand_output)
            continue

        unavailable.append(command["command"])
        lines.extend(["Parser help was unavailable in the generating environment:", ""])
        add_help_block(lines, output)

    if unavailable and not allow_unavailable:
        lines.append(".. unavailable command help is treated as a generation failure.")
    return "\n".join(lines), unavailable


def main() -> int:
    """Generate the reference and optionally fail for unavailable command help."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python runtime for CLI help")
    parser.add_argument("--timeout", type=int, default=60, help="Per-command help timeout")
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Write diagnostic sections instead of failing for unavailable parser help",
    )
    parser.add_argument("--check", action="store_true", help="Fail if generated output is stale")
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of concurrent help processes (default: 4).",
    )
    parser.add_argument(
        "--script",
        action="append",
        dest="scripts",
        metavar="SCRIPT",
        help="Generate only this module stem; may be supplied more than once.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include commands marked as requiring optional CLI dependencies.",
    )
    args = parser.parse_args()

    inventory_scripts = {
        command["script"]
        for command in json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))["commands"]
    }
    selected_scripts = set(args.scripts) if args.scripts else None
    if selected_scripts and (unknown := selected_scripts - inventory_scripts):
        parser.error("unknown inventory script(s): " + ", ".join(sorted(unknown)))

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    content, unavailable = render(
        args.python,
        args.timeout,
        args.allow_unavailable,
        selected_scripts,
        args.jobs,
        args.include_optional,
    )
    if args.check:
        existing = (
            OUTPUT_PATH.read_text(encoding="utf-8")
            if OUTPUT_PATH.is_file()
            else ""
        )
        if normalize_for_comparison(existing) != normalize_for_comparison(content):
            print("CLI reference is stale; rerun docs/generate_cli_reference.py", file=sys.stderr)
            diff = difflib.unified_diff(
                existing.splitlines(),
                content.splitlines(),
                fromfile=str(OUTPUT_PATH),
                tofile="regenerated CLI reference",
                n=2,
            )
            diff_lines = list(diff)
            print("\n".join(diff_lines[:200]), file=sys.stderr)
            if len(diff_lines) > 200:
                print(
                    f"... diff truncated ({len(diff_lines) - 200} more lines)",
                    file=sys.stderr,
                )
            return 1
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(content, encoding="utf-8")

    if unavailable:
        print(
            "CLI help unavailable for: " + ", ".join(unavailable),
            file=sys.stderr,
        )
        return 0 if args.allow_unavailable else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
