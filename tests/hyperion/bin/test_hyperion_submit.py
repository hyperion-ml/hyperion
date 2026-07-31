"""Tests for the scheduler-neutral recipe submitter."""

import os
import sys
from pathlib import Path

import pytest

from hyperion.bin.hyperion_submit import (
    ArraySpec,
    SubmitOptions,
    _command_with_torchrun,
    _make_submit_options,
    _slurm_fallback_log_paths,
    _slurm_options,
    render_slurm_script,
    run_local,
    run_slurm,
)


def make_options(tmp_path: Path, **kwargs) -> SubmitOptions:
    """Create baseline portable options for a test."""
    values = {
        "command": ["command"],
        "output_file": tmp_path / "job.log",
        "num_gpus": 0,
        "num_threads": 1,
        "mem": None,
        "mem_per_cpu": None,
        "time_limit": None,
        "array": None,
        "max_jobs_run": None,
    }
    values.update(kwargs)
    return SubmitOptions(**values)


def test_array_spec_parsing() -> None:
    """Array specifications validate names and inclusive ranges."""
    assert ArraySpec.parse("JOB=1:3") == ArraySpec("JOB", 1, 3)
    with pytest.raises(ValueError, match="form"):
        ArraySpec.parse("JOB=1")
    with pytest.raises(ValueError, match="bounds"):
        ArraySpec.parse("JOB=3:1")


def test_submit_options_validate_memory_scope(tmp_path: Path) -> None:
    """Flat portable options are retained and conflicting memory scopes fail."""
    options = _make_submit_options(
        {
            "command": ["echo", "hello"],
            "output_file": str(tmp_path / "job.log"),
            "num_gpus": 1,
            "num_threads": 2,
            "mem": None,
            "mem_per_cpu": "4G",
            "time": None,
            "array": None,
            "max_jobs_run": None,
        }
    )
    assert options.num_gpus == 1
    assert options.num_threads == 2
    assert options.mem_per_cpu == "4G"

    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_submit_options(
            {
                "command": ["echo"],
                "output_file": str(tmp_path / "job.log"),
                "num_gpus": 0,
                "num_threads": 1,
                "mem": "8G",
                "mem_per_cpu": "1G",
                "time": None,
                "array": None,
                "max_jobs_run": None,
            }
        )


def test_parser_loads_flat_yaml_and_preserves_command_boundary(tmp_path: Path) -> None:
    """The jsonargparse config format supplies flat portable options."""
    from hyperion.bin.hyperion_submit import make_parser

    config_file = tmp_path / "submit.yaml"
    config_file.write_text(
        "num_gpus: 2\nnum_threads: 4\nmem_per_cpu: 3G\nbase_options:\n  - --nodes=1\n",
        encoding="utf-8",
    )
    args = make_parser().parse_args(
        [
            "local",
            "--cfg",
            str(config_file),
            "--output-file",
            str(tmp_path / "job.log"),
            "--",
            "echo",
            "hello",
        ]
    )
    options = _make_submit_options(args.as_dict()["local"])

    assert options.command == ["echo", "hello"]
    assert options.num_gpus == 2
    assert options.num_threads == 4
    assert options.mem_per_cpu == "3G"


def test_base_options_cannot_override_submitter_options(tmp_path: Path) -> None:
    """Site policy cannot override a resource value owned by the submitter."""
    with pytest.raises(ValueError, match="base_options"):
        _slurm_options(
            make_options(tmp_path),
            {"base_options": ["--time=24:00:00"]},
        )


def test_local_array_substitutes_job_and_redirects_logs(tmp_path: Path) -> None:
    """Local arrays run sequentially with task-specific arguments and logs."""
    script = tmp_path / "record_task.py"
    script.write_text(
        "import os, sys\nprint(sys.argv[1])\nprint(os.environ['JOB'])\n",
        encoding="utf-8",
    )
    options = make_options(
        tmp_path,
        command=[sys.executable, str(script), "JOB"],
        output_file=tmp_path / "logs" / "task.JOB.log",
        array=ArraySpec("JOB", 1, 2),
    )

    run_local(options)

    assert (tmp_path / "logs" / "task.1.log").read_text().splitlines() == ["1", "1"]
    assert (tmp_path / "logs" / "task.2.log").read_text().splitlines() == ["2", "2"]


def test_multi_gpu_command_uses_torchrun_and_resolves_entrypoint(monkeypatch) -> None:
    """Multi-GPU commands use torchrun with a PATH-resolved entry point."""
    monkeypatch.setattr(
        "hyperion.bin.hyperion_submit.shutil.which",
        lambda command: f"/env/bin/{command}",
    )
    assert _command_with_torchrun(["program"], 1) == ["program"]
    assert _command_with_torchrun(["program"], 2) == [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "/env/bin/program",
    ]


def test_slurm_script_logs_environment_and_expands_array(
    tmp_path: Path, monkeypatch
) -> None:
    """Rendered scripts retain the legacy diagnostics without Conda wrapping."""
    monkeypatch.setattr(
        "hyperion.bin.hyperion_submit.shutil.which",
        lambda command: f"/env/bin/{command}",
    )
    options = make_options(
        tmp_path,
        command=["program", "--part-idx", "JOB", "out.JOB.ark"],
        output_file=tmp_path / "logs" / "job.JOB.log",
        num_gpus=2,
        num_threads=4,
        array=ArraySpec("JOB", 1, 3),
    )
    script = render_slurm_script(options, tmp_path)

    assert 'export JOB="${SLURM_ARRAY_TASK_ID}"' in script
    assert "env | sort | grep '^SLURM_'" in script
    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}" in script
    assert "exec > " in script
    assert "exec >> " not in script
    assert "torchrun --standalone --nnodes=1 --nproc-per-node=2" in script
    assert '"${JOB}"' in script
    assert "conda activate" not in script


def test_slurm_options_translate_resources_and_array_limit(tmp_path: Path) -> None:
    """Portable resources become the expected sbatch arguments."""
    options = make_options(
        tmp_path,
        num_gpus=2,
        num_threads=8,
        mem="32G",
        time_limit="24:00:00",
        array=ArraySpec("JOB", 1, 10),
        max_jobs_run=3,
    )
    args = _slurm_options(
        options,
        {
            "base_options": ["--nodes=1"],
            "default_gpu_type": "v100",
            "gpu_types": {"v100": {"options": ["--gres=gpu:v100:{num_gpus}"]}},
        },
    )
    assert "--export=ALL" in args
    assert "--cpus-per-task=8" in args
    assert "--mem=32G" in args
    assert "--time=24:00:00" in args
    assert "--array=1-10%3" in args
    assert "--gres=gpu:v100:2" in args


def test_slurm_fallback_logs_are_per_array_task(tmp_path: Path) -> None:
    """Scheduler diagnostics use Slurm filename tokens outside the submit directory."""
    stdout, stderr = _slurm_fallback_log_paths(
        tmp_path / "logs" / "extract.JOB.log", ArraySpec("JOB", 1, 10)
    )
    assert stdout == tmp_path / "logs" / "q" / "slurm-%A_%a.out"
    assert stderr == tmp_path / "logs" / "q" / "slurm-%A_%a.err"


def test_run_slurm_uses_stub_scheduler_commands(tmp_path: Path, monkeypatch) -> None:
    """Slurm execution submits and waits without a real Slurm installation."""
    options = make_options(tmp_path, output_file=tmp_path / "logs" / "job.log")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    submitted_args = tmp_path / "sbatch.args"
    scripts = {
        "sbatch": "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$STUB_ARGS\"\nprintf '12345;cluster\\n'\n",
        "squeue": "#!/bin/sh\nexit 0\n",
        "sacct": "#!/bin/sh\nprintf 'COMPLETED|0:0\\n'\n",
    }
    for name, content in scripts.items():
        path = bin_dir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o700)
    monkeypatch.setenv("STUB_ARGS", str(submitted_args))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{Path.cwd()}")

    run_slurm(options, {"sbatch_command": "sbatch", "cpu_options": ["--partition=cpu"]})

    sbatch_args = submitted_args.read_text(encoding="utf-8").splitlines()
    script_path = Path(sbatch_args[-1])
    assert script_path.exists()
    assert "--partition=cpu" in sbatch_args
    assert f"--output={tmp_path / 'logs' / 'q' / 'slurm-%j.out'}" in sbatch_args
    assert f"--error={tmp_path / 'logs' / 'q' / 'slurm-%j.err'}" in sbatch_args
