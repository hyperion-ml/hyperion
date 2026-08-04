#!/usr/bin/env python
"""Convert trusted legacy PyTorch checkpoints to Hyperion model directories."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
from jsonargparse import ActionYesNo, ArgumentParser

from hyperion.hyp_defs import config_logger
from hyperion.torch import HyperTorchModel
from hyperion.torch.trainers.torch_trainer_base import TorchTrainerBase


def _trainer_state_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract shared trainer state from a legacy modern-trainer checkpoint.

    Args:
        checkpoint: Legacy serialized checkpoint dictionary.

    Returns:
        Shared trainer-state dictionary suitable for ``trainer_state.json``.

    Raises:
        ValueError: If the checkpoint cannot resume a modern trainer.
    """
    required_keys = ("epoch", "rng_state", "optimizer_state_dict")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            "Checkpoint does not contain the modern trainer state required for "
            f"conversion: {', '.join(missing_keys)}"
        )

    if "step" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'step' and cannot resume a "
            "TorchTrainerBase trainer."
        )

    return {
        key: checkpoint[key]
        for key in ("epoch", "batch", "step", "logs")
        if key in checkpoint
    }


def _normalized_json(value: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-normalized state dictionary for comparison and writing."""
    return json.loads(json.dumps(value, default=HyperTorchModel._json_default))


def _save_model_checkpoint(
    output_dir: Path, model_name: str, checkpoint: Dict[str, Any], save_state: bool
) -> None:
    """Write one converted model directory atomically.

    Args:
        output_dir: New checkpoint root directory.
        model_name: Name of the model subdirectory.
        checkpoint: Legacy checkpoint dictionary.
        save_state: Whether to include optimizer, scheduler, and SWA state.

    Raises:
        FileExistsError: If the destination model directory already exists.
    """
    model_dir = output_dir / model_name
    if model_dir.exists():
        raise FileExistsError(f"Output model directory already exists: {model_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{model_name}.tmp-", dir=output_dir))
    try:
        HyperTorchModel.save_config_state_dict(
            tmp_dir,
            checkpoint["model_cfg"],
            checkpoint["model_state_dict"],
        )
        if save_state:
            torch.save(
                checkpoint["optimizer_state_dict"],
                tmp_dir / TorchTrainerBase.optimizer_state_file_name,
            )
            state_files = (
                ("lr_scheduler_state_dict", TorchTrainerBase.lr_scheduler_state_file_name),
                ("wd_scheduler_state_dict", TorchTrainerBase.wd_scheduler_state_file_name),
                ("swa_scheduler_state_dict", TorchTrainerBase.swa_scheduler_state_file_name),
            )
            for state_key, file_name in state_files:
                if state_key in checkpoint:
                    torch.save(checkpoint[state_key], tmp_dir / file_name)

            if "swa_model_state_dict" in checkpoint:
                from safetensors.torch import save_file

                save_file(
                    HyperTorchModel.prepare_safetensors_state_dict(
                        checkpoint["swa_model_state_dict"]
                    ),
                    str(tmp_dir / TorchTrainerBase.swa_model_weights_file_name),
                )

        tmp_dir.replace(model_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _validate_trainer_state(
    output_dir: Path, checkpoint: Dict[str, Any]
) -> bool:
    """Validate shared state against an existing conversion when present.

    Args:
        output_dir: New checkpoint root directory.
        checkpoint: Legacy checkpoint dictionary.

    Raises:
        ValueError: If existing shared state conflicts with ``checkpoint``.
    """
    trainer_state = _normalized_json(_trainer_state_from_checkpoint(checkpoint))
    trainer_state_path = output_dir / TorchTrainerBase.trainer_state_file_name
    rng_state_path = output_dir / TorchTrainerBase.rng_state_file_name

    if trainer_state_path.exists() != rng_state_path.exists():
        raise ValueError(
            "Checkpoint root contains incomplete shared trainer state: "
            f"{output_dir}"
        )

    if trainer_state_path.is_file():
        with trainer_state_path.open("r", encoding="utf-8") as f:
            existing_trainer_state = json.load(f)
        existing_rng_state = torch.load(
            rng_state_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        if (
            existing_trainer_state != trainer_state
            or not torch.equal(existing_rng_state, checkpoint["rng_state"])
        ):
            raise ValueError(
                "Shared trainer state conflicts with the existing checkpoint root: "
                f"{output_dir}"
            )
        return True

    return False


def _write_trainer_state(output_dir: Path, checkpoint: Dict[str, Any]) -> None:
    """Write shared trainer state for a converted checkpoint.

    Args:
        output_dir: New checkpoint root directory.
        checkpoint: Legacy checkpoint dictionary.
    """
    trainer_state = _normalized_json(_trainer_state_from_checkpoint(checkpoint))
    trainer_state_path = output_dir / TorchTrainerBase.trainer_state_file_name
    rng_state_path = output_dir / TorchTrainerBase.rng_state_file_name

    with trainer_state_path.open("w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)
        f.write("\n")
    torch.save(checkpoint["rng_state"], rng_state_path)


def convert_checkpoint(
    in_model_file: Path,
    out_model_dir: Path,
    model_name: str = "model",
    get_trainer_state: bool = False,
) -> None:
    """Convert a trusted legacy model checkpoint to the modern directory format.

    Args:
        in_model_file: Trusted legacy ``.pth`` checkpoint file.
        out_model_dir: New checkpoint root directory.
        model_name: Name of the output model subdirectory.
        get_trainer_state: Whether to migrate full trainer-resume state.
    """
    if not in_model_file.is_file():
        raise FileNotFoundError(f"Input model file not found: {in_model_file}")
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be a single non-empty directory name")

    logging.warning(
        "Loading legacy checkpoint with weights_only=False. Only convert trusted "
        "checkpoint files."
    )
    checkpoint = torch.load(
        in_model_file,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    required_keys = ("model_cfg", "model_state_dict")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            "Input checkpoint does not contain a Hyperion model: "
            f"{', '.join(missing_keys)}"
        )

    out_model_dir.mkdir(parents=True, exist_ok=True)
    has_trainer_state = (
        _validate_trainer_state(out_model_dir, checkpoint)
        if get_trainer_state
        else False
    )
    _save_model_checkpoint(
        out_model_dir, model_name, checkpoint, save_state=get_trainer_state
    )
    if get_trainer_state:
        try:
            if not has_trainer_state:
                _write_trainer_state(out_model_dir, checkpoint)
        except Exception:
            shutil.rmtree(out_model_dir / model_name, ignore_errors=True)
            raise

    logging.info("Converted %s to %s", in_model_file, out_model_dir / model_name)


def make_parser() -> ArgumentParser:
    """Create the command-line parser for legacy checkpoint conversion.

    Returns:
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Convert a trusted legacy Hyperion .pth model checkpoint."
    )
    parser.add_argument(
        "--in-model-file",
        required=True,
        type=Path,
        help="Trusted legacy .pth model checkpoint.",
    )
    parser.add_argument(
        "--out-model-dir",
        required=True,
        type=Path,
        help="Output checkpoint root directory.",
    )
    parser.add_argument(
        "--model-name",
        default="model",
        help="Model subdirectory name within --out-model-dir.",
    )
    parser.add_argument(
        "--get-trainer-state",
        default=False,
        action=ActionYesNo,
        help="Also convert optimizer, scheduler, SWA, and trainer resume state.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Logging verbosity level.",
    )
    return parser


def main() -> None:
    """Run the legacy checkpoint conversion command."""
    args = make_parser().parse_args()
    config_logger(args.verbose)
    del args.verbose
    convert_checkpoint(**vars(args))


if __name__ == "__main__":
    main()
