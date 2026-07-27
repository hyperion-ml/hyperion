"""Submit Hyperion recipe commands locally or through Slurm.

The launcher deliberately uses the environment inherited from its caller.  In
particular, it neither activates Conda nor selects CUDA devices: Slurm assigns
GPUs and exports the selected ``CUDA_VISIBLE_DEVICES`` to batch jobs.
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jsonargparse import ActionConfigFile, ArgumentParser

_ARRAY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(\d+):(\d+)$")
_RESERVED_BASE_OPTION_PREFIXES = (
    "--array",
    "--cpus-per-task",
    "--error",
    "--export",
    "--gpus",
    "--gres",
    "--mem",
    "--mem-per-cpu",
    "--mem-per-gpu",
    "--output",
    "--parsable",
    "--time",
)

@dataclass(frozen=True)
class ArraySpec:
    """Inclusive task range and task-variable name.

    Attributes:
        name: Environment variable available to each task.
        start: First task index.
        end: Last task index.
    """

    name: str
    start: int
    end: int

    @classmethod
    def parse(cls, value: str | None) -> "ArraySpec | None":
        """Parse an array specification.

        Args:
            value: Specification in ``NAME=START:END`` form.

        Returns:
            The parsed specification, or ``None`` when no array was requested.

        Raises:
            ValueError: If the specification is malformed or has an invalid range.
        """
        if value is None:
            return None
        match = _ARRAY_RE.fullmatch(value)
        if match is None:
            raise ValueError("--array must have the form NAME=START:END")
        name, start, end = match.groups()
        start_i, end_i = int(start), int(end)
        if start_i <= 0 or start_i > end_i:
            raise ValueError("array bounds must be positive and START must not exceed END")
        return cls(name=name, start=start_i, end=end_i)

    def values(self) -> range:
        """Return the task indices in ascending order."""
        return range(self.start, self.end + 1)


@dataclass(frozen=True)
class SubmitOptions:
    """Portable resource and command options shared by all backends.

    Attributes:
        command: Target executable and its arguments.
        output_file: Combined stdout/stderr log file.
        num_gpus: GPUs requested per task.
        num_threads: CPU threads requested per task.
        mem: Total memory requested per node.
        mem_per_cpu: Memory requested per allocated CPU.
        time_limit: Scheduler wall-clock limit.
        array: Optional array specification.
        max_jobs_run: Maximum simultaneously running array tasks.
    """

    command: list[str]
    output_file: Path
    num_gpus: int
    num_threads: int
    mem: str | None
    mem_per_cpu: str | None
    time_limit: str | None
    array: ArraySpec | None
    max_jobs_run: int | None


def _configure_logging(verbose: int) -> None:
    """Configure submitter logging without importing the ML runtime.

    Args:
        verbose: Verbosity level in the shared Hyperion CLI convention.
    """
    levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=levels[verbose], format="%(levelname)s: %(message)s")


def _make_submit_options(args: dict[str, Any]) -> SubmitOptions:
    """Validate parsed portable submission options.

    Args:
        args: Parsed backend arguments as a dictionary.

    Returns:
        Validated submission options.

    Raises:
        ValueError: If an option combination is invalid.
    """
    mem = args.get("mem")
    mem_per_cpu = args.get("mem_per_cpu")
    if mem is not None and mem_per_cpu is not None:
        raise ValueError("--mem and --mem-per-cpu are mutually exclusive")

    num_gpus = args.get("num_gpus", 0)
    num_threads = args.get("num_threads", 1)
    time_limit = args.get("time")

    if num_gpus < 0:
        raise ValueError("--num-gpus must be non-negative")
    if num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    max_jobs_run = args.get("max_jobs_run")
    array = ArraySpec.parse(args.get("array"))
    if max_jobs_run is not None:
        if array is None:
            raise ValueError("--max-jobs-run requires --array")
        if max_jobs_run <= 0:
            raise ValueError("--max-jobs-run must be positive")

    command = list(args.get("command", []))
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    output_file = Path(args["output_file"])
    if array is not None and array.name not in str(output_file):
        raise ValueError("array output files must contain the array variable name")

    return SubmitOptions(
        command=command,
        output_file=output_file,
        num_gpus=num_gpus,
        num_threads=num_threads,
        mem=mem,
        mem_per_cpu=mem_per_cpu,
        time_limit=time_limit,
        array=array,
        max_jobs_run=max_jobs_run,
    )


def _substitute_task(value: str, array: ArraySpec | None, task_id: int | None) -> str:
    """Substitute an array task variable with a concrete task index."""
    if array is None:
        return value
    assert task_id is not None
    return value.replace(array.name, str(task_id))


def _task_command(options: SubmitOptions, task_id: int | None) -> list[str]:
    """Return the concrete command for a local task."""
    return [_substitute_task(arg, options.array, task_id) for arg in options.command]


def _task_output_file(options: SubmitOptions, task_id: int | None) -> Path:
    """Return the concrete output path for a local task."""
    return Path(_substitute_task(str(options.output_file), options.array, task_id))


def _command_with_torchrun(command: Sequence[str], num_gpus: int) -> list[str]:
    """Prefix a command with torchrun for multi-GPU execution.

    Args:
        command: Target command and its arguments.
        num_gpus: GPUs allocated to the command.

    Returns:
        The command, optionally prefixed with torchrun.
    """
    if num_gpus <= 1:
        return list(command)
    return [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={num_gpus}",
        *command,
    ]


def run_local(options: SubmitOptions) -> None:
    """Run a command locally and synchronously.

    Args:
        options: Validated submission options.

    Raises:
        subprocess.CalledProcessError: If a task exits unsuccessfully.
    """
    task_ids: Sequence[int | None] = (
        list(options.array.values()) if options.array is not None else [None]
    )
    for task_id in task_ids:
        output_file = _task_output_file(options, task_id)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        command = _command_with_torchrun(_task_command(options, task_id), options.num_gpus)
        env = os.environ.copy()
        if options.array is not None:
            assert task_id is not None
            env[options.array.name] = str(task_id)
        logging.info("running locally: %s", shlex.join(command))
        with output_file.open("w", encoding="utf-8") as log_file:
            subprocess.run(command, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def _shell_quote_with_array(value: str, array: ArraySpec | None) -> str:
    """Quote one shell argument while preserving array-variable expansion."""
    if array is None or array.name not in value:
        return shlex.quote(value)
    parts = value.split(array.name)
    quoted_parts = [shlex.quote(part) for part in parts]
    variable = f'"${{{array.name}}}"'
    return variable.join(quoted_parts)


def _shell_command(command: Sequence[str], array: ArraySpec | None) -> str:
    """Render an argument-safe shell command with optional task expansion."""
    return " ".join(_shell_quote_with_array(arg, array) for arg in command)


def _slurm_script_path(output_file: Path) -> Path:
    """Return the durable batch-script path associated with an output log."""
    return output_file.parent / "q" / f"{output_file.stem}.submit.sh"


def render_slurm_script(options: SubmitOptions, cwd: Path) -> str:
    """Render the batch script executed by Slurm.

    Args:
        options: Validated submission options.
        cwd: Directory in which the recipe command must execute.

    Returns:
        Complete bash batch script content.
    """
    output_file = _shell_quote_with_array(str(options.output_file), options.array)
    command = _shell_command(
        _command_with_torchrun(options.command, options.num_gpus), options.array
    )
    array_setup = ""
    if options.array is not None:
        array_setup = f"export {options.array.name}=\"${{SLURM_ARRAY_TASK_ID}}\"\n"
    return f"""#!/usr/bin/env bash
cd {shlex.quote(str(cwd))}
{array_setup}mkdir -p \"$(dirname {output_file})\"
exec >> {output_file} 2>&1
echo \"# Running on $(hostname)\"
echo \"# Started at $(date)\"
env | sort | grep '^SLURM_' | while IFS= read -r line; do echo \"# $line\"; done
echo \"# CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES-}}\"
echo \"# {command}\"
begin_time=$(date +%s)
{command}
ret=$?
end_time=$(date +%s)
echo \"# Accounting: begin_time=$begin_time\"
echo \"# Accounting: end_time=$end_time\"
echo \"# Accounting: time=$((end_time-begin_time)) threads={options.num_threads}\"
echo \"# Finished at $(date) with status $ret\"
exit $ret
"""


def _slurm_options(options: SubmitOptions, config: dict[str, Any]) -> list[str]:
    """Translate portable resource requests to Slurm arguments."""
    result = ["--export=ALL", "--parsable"]
    base_options = config.get("base_options", [])
    for option in base_options:
        if option.startswith(_RESERVED_BASE_OPTION_PREFIXES):
            raise ValueError(f"base_options cannot override submitter option: {option}")
    result.extend(base_options)
    result.append(f"--cpus-per-task={options.num_threads}")
    if options.mem is not None:
        result.append(f"--mem={options.mem}")
    if options.mem_per_cpu is not None:
        result.append(f"--mem-per-cpu={options.mem_per_cpu}")
    if options.time_limit is not None:
        result.append(f"--time={options.time_limit}")
    if options.array is not None:
        array_arg = f"{options.array.start}-{options.array.end}"
        if options.max_jobs_run is not None:
            array_arg += f"%{options.max_jobs_run}"
        result.append(f"--array={array_arg}")
    if options.num_gpus > 0:
        gpu_types = config.get("gpu_types", {}) or {}
        gpu_type = config.get("default_gpu_type")
        if gpu_type not in gpu_types:
            raise ValueError("Slurm configuration has no valid default_gpu_type")
        for option in gpu_types[gpu_type].get("options", []):
            try:
                result.append(option.format(num_gpus=options.num_gpus))
            except KeyError as exc:
                raise ValueError(f"unsupported placeholder in Slurm option: {option}") from exc
    else:
        result.extend(config.get("cpu_options", []))
    return result


def _parse_job_id(sbatch_output: str) -> str:
    """Extract a Slurm job id from ``sbatch --parsable`` output."""
    job_id = sbatch_output.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse Slurm job id from: {sbatch_output!r}")
    return job_id


def wait_for_slurm_job(job_id: str, poll_interval: float = 2.0) -> None:
    """Wait for a Slurm job and raise if accounting reports a failed task.

    Args:
        job_id: Parent Slurm job ID.
        poll_interval: Seconds between active-job checks.

    Raises:
        RuntimeError: If Slurm accounting reports a failed job or task.
    """
    while True:
        queued = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
            check=True,
            capture_output=True,
            text=True,
        )
        if not queued.stdout.strip():
            break
        time.sleep(poll_interval)

    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            job_id,
            "--format=State,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [line.split("|", maxsplit=1) for line in accounting.stdout.splitlines() if line]
    if not states:
        raise RuntimeError(f"Slurm accounting returned no records for job {job_id}")
    failures = []
    for line in states:
        state = line[0].split()[0]
        exit_code = line[1] if len(line) > 1 else ""
        if state != "COMPLETED" or not exit_code.startswith("0:"):
            failures.append(line)
    if failures:
        raise RuntimeError(f"Slurm job {job_id} failed: {', '.join('|'.join(x) for x in failures)}")


def run_slurm(options: SubmitOptions, config: dict[str, Any]) -> None:
    """Submit a Slurm batch script and wait for completion.

    Args:
        options: Validated submission options.
        config: Parsed submission configuration.
    """
    output_file = options.output_file
    script_output = (
        Path(_substitute_task(str(output_file), options.array, options.array.start))
        if options.array is not None
        else output_file
    )
    script_path = _slurm_script_path(script_output)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_slurm_script(options, Path.cwd()), encoding="utf-8")
    script_path.chmod(0o700)

    command = [str(config.get("sbatch_command", "sbatch"))]
    command.extend(_slurm_options(options, config))
    command.append(str(script_path.resolve()))
    logging.info("submitting Slurm job: %s", shlex.join(command))
    submitted = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = _parse_job_id(submitted.stdout)
    logging.info("submitted Slurm job %s", job_id)
    wait_for_slurm_job(job_id)


def _add_common_args(parser: ArgumentParser) -> None:
    """Add portable submitter arguments to a backend parser."""
    parser.add_argument("--cfg", action=ActionConfigFile, help="YAML submitter config")
    parser.add_argument("--output-file", required=True, help="combined stdout/stderr log")
    parser.add_argument("--num-gpus", type=int, default=0, help="GPUs per task")
    parser.add_argument("--num-threads", type=int, default=1, help="CPU threads per task")
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--mem", default=None, help="total memory per node")
    memory.add_argument("--mem-per-cpu", default=None, help="memory per allocated CPU")
    parser.add_argument("--time", default=None, help="wall-clock limit")
    parser.add_argument("--array", default=None, help="array range NAME=START:END")
    parser.add_argument(
        "--max-jobs-run", type=int, default=None, help="maximum running array tasks"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2, 3], default=1)


def _add_slurm_config_args(parser: ArgumentParser) -> None:
    """Add flat Slurm site-policy configuration fields.

    Args:
        parser: Parser accepting a submitter YAML configuration.
    """
    parser.add_argument("--sbatch-command", default="sbatch")
    parser.add_argument("--base-options", type=list, default=[])
    parser.add_argument("--cpu-options", type=list, default=[])
    parser.add_argument("--gpu-types", type=dict, default={})
    parser.add_argument("--default-gpu-type", default=None)


def make_parser() -> ArgumentParser:
    """Build the ``hyperion-submit`` command-line parser.

    Returns:
        Configured parser with local and Slurm subcommands.
    """
    parser = ArgumentParser(description="Run Hyperion recipe commands locally or through Slurm")
    subcommands = parser.add_subcommands(required=True)
    local_parser = ArgumentParser(prog="hyperion-submit local")
    slurm_parser = ArgumentParser(prog="hyperion-submit slurm")
    _add_common_args(local_parser)
    _add_common_args(slurm_parser)
    _add_slurm_config_args(local_parser)
    _add_slurm_config_args(slurm_parser)
    subcommands.add_subcommand("local", local_parser)
    subcommands.add_subcommand("slurm", slurm_parser)
    return parser


def main() -> None:
    """Parse arguments and execute the selected backend."""
    parser = make_parser()
    if "--" not in sys.argv[1:] and not {"-h", "--help"}.intersection(sys.argv[1:]):
        parser.error("the target command must be separated from submitter options with --")
    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = args.as_dict()[subcommand]
    _configure_logging(kwargs["verbose"])
    try:
        options = _make_submit_options(kwargs)
        if subcommand == "local":
            run_local(options)
        else:
            run_slurm(options, kwargs)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        logging.error("submission failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
