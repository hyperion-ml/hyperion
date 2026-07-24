"""Representative end-to-end smoke tests for stable Hyperion CLI commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hyperion.utils import TrialKey, TrialScores

REPO_DIR = Path(__file__).resolve().parents[2]
KEY_FIXTURE = REPO_DIR / "tests" / "data_in" / "core-core_det5_key.h5"


def run_cli(*arguments: str) -> None:
    """Run one installed-module CLI from the repository root."""
    subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=REPO_DIR,
        check=True,
        text=True,
        capture_output=True,
    )


def make_scores(output_file: Path) -> TrialScores:
    """Create deterministic fixture scores aligned with the shared trial key."""
    key = TrialKey.load(KEY_FIXTURE)
    mask = np.logical_or(key.tar, key.non)
    scores = np.arange(mask.size, dtype=float).reshape(mask.shape) * mask
    trial_scores = TrialScores(key.model_set, key.seg_set, scores, mask)
    trial_scores.save(output_file)
    return trial_scores


def test_table_cat_command(tmp_path: Path) -> None:
    """The data-preparation table utility concatenates CSV manifests."""
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "combined.csv"
    pd.DataFrame({"id": ["utt1"], "speaker": ["spk1"]}).to_csv(first, index=False)
    pd.DataFrame({"id": ["utt2"], "speaker": ["spk2"]}).to_csv(second, index=False)

    run_cli(
        "hyperion.bin.hyperion_tables",
        "cat",
        "--table-type",
        "generic",
        "--input-files",
        str(first),
        str(second),
        "--output-file",
        str(output),
    )

    assert pd.read_csv(output).id.tolist() == ["utt1", "utt2"]


def test_vad_format_round_trip_command(tmp_path: Path) -> None:
    """The conversion CLI writes binary VAD then returns a timestamp manifest."""
    time_marks = tmp_path / "utt1.csv"
    input_manifest = tmp_path / "input-vad.csv"
    segments = tmp_path / "segments.csv"
    binary_ark = tmp_path / "vad.ark"
    binary_csv = tmp_path / "vad.csv"
    output_manifest = tmp_path / "output-vad.csv"
    output_dir = tmp_path / "time-marks"

    pd.DataFrame({"start": [0.10, 0.60], "end": [0.35, 0.80]}).to_csv(
        time_marks, index=False
    )
    pd.DataFrame({"id": ["utt1"], "storage_path": [time_marks]}).to_csv(
        input_manifest, index=False
    )
    pd.DataFrame(
        {"id": ["utt1"], "recording_id": ["rec1"], "start": [0.0], "duration": [1.0]}
    ).to_csv(segments, index=False)

    run_cli(
        "hyperion.bin.convert_vad_format",
        "time_marks_to_bin",
        "--in-vad-file",
        str(input_manifest),
        "--out-vad-file",
        f"ark,csv:{binary_ark},{binary_csv}",
        "--segments-file",
        str(segments),
    )
    run_cli(
        "hyperion.bin.convert_vad_format",
        "bin_to_time_marks",
        "--in-vad-file",
        f"csv:{binary_csv}",
        "--out-vad-file",
        str(output_manifest),
        "--output-dir",
        str(output_dir),
    )

    assert output_manifest.is_file()
    restored = pd.read_csv(output_dir / "utt1.csv")
    assert list(restored.columns) == ["start", "end"]
    assert not restored.empty


def test_verification_evaluation_and_score_merge_commands(tmp_path: Path) -> None:
    """Scoring utility and verification evaluator produce readable artifacts."""
    scores_file = tmp_path / "scores.h5"
    metrics_file = tmp_path / "metrics.csv"
    merged_file = tmp_path / "merged.h5"
    scores = make_scores(scores_file)

    run_cli(
        "hyperion.bin.eval_verification_metrics",
        "--key-files",
        str(KEY_FIXTURE),
        "--key-names",
        "fixture",
        "--score-files",
        str(scores_file),
        "--score-names",
        "linear",
        "--output-file",
        str(metrics_file),
    )
    metrics = pd.read_csv(metrics_file)
    assert {"scores", "key", "eer"}.issubset(metrics.columns)

    parts: list[Path] = []
    for enroll_part in range(1, 3):
        for test_part in range(1, 3):
            part = tmp_path / f"scores.{enroll_part}.{test_part}.h5"
            scores.split(enroll_part, 2, test_part, 2).save(part)
            parts.append(part)
    run_cli(
        "hyperion.bin.merge_scores",
        "--input-files",
        *(str(part) for part in parts),
        "--output-file",
        str(merged_file),
    )
    merged_scores = TrialScores.load(merged_file)
    merged_scores.sort()
    expected_scores = scores.copy()
    expected_scores.sort()
    assert merged_scores == expected_scores
