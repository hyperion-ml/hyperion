#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import time
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.augment import SpeechAugment
from hyperion.np.preprocessing import ResamplerToTargetFreq
from hyperion.torch import TorchModel
from hyperion.torch.utils import open_device
from hyperion.utils import HyperDataset


def init_device(use_gpu: bool) -> torch.device:
    """Initialise execution device based on user preference.

    Args:
        use_gpu: When ``True`` the model is moved to GPU (if available).

    Returns:
        torch.device: Device where inference will run.
    """
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus=%d", num_gpus)
    device = open_device(num_gpus=num_gpus)
    return device


def load_model(model_path: str, device: torch.device) -> TorchModel:
    """Load a serialized TorchModel and move it to ``device``.

    Args:
        model_path: File path containing the serialized model.
        device: Destination device where the model will be placed.

    Returns:
        TorchModel: Loaded model ready for inference.
    """
    logging.info("loading model %s", model_path)
    model = TorchModel.auto_load(model_path)
    logging.info("qvector-model=%s", model)
    model.to(device)
    model.eval()
    return model


def augment(
    key0: str,
    x0: np.ndarray,
    augmenter: Optional[SpeechAugment],
    aug_df: Optional[List[pd.DataFrame]],
    aug_id: int,
) -> Tuple[str, np.ndarray]:
    """Apply optional augmentation to the waveform.

    Args:
        key0: Original utterance identifier.
        x0: Original waveform (channels, samples).
        augmenter: Configured augmenter or ``None``.
        aug_df: List used to accumulate augmentation metadata.
        aug_id: Augmentation index for logging.

    Returns:
        Tuple[str, np.ndarray]: Augmented key and waveform.
    """
    if augmenter is None:
        x = x0
        key = key0
    else:
        x, aug_info = augmenter(x0)
        key = "%s-aug-%02d" % (key0, aug_id)
        aug_df_row = {
            "key_aug": key,
            "key_orig": key0,
            "noise_type": aug_info["noise"]["noise_type"],
            "snr": aug_info["noise"]["snr"],
            "rir_type": aug_info["reverb"]["rir_type"],
            "srr": aug_info["reverb"]["srr"],
            "sdr": aug_info["sdr"],
        }

        if aug_df is not None:
            aug_df.append(pd.DataFrame(aug_df_row, index=[0]))

    return key, x


def select_random_chunk(
    key: str,
    x: torch.Tensor,
    fs: int,
    min_utt_length: float,
    max_utt_length: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Select a random chunk within the waveform.

    Args:
        key: Utterance identifier (used only for logging).
        x: Waveform tensor shaped (1, time).
        fs: Sampling frequency.
        min_utt_length: Minimum chunk length in seconds.
        max_utt_length: Maximum chunk length in seconds.
        rng: Random generator instance.

    Returns:
        torch.Tensor: Cropped chunk; original tensor if no cropping applied.
    """
    utt_length = rng.integers(
        low=int(fs * min_utt_length), high=int(fs * max_utt_length + 1)
    )
    if utt_length < x.shape[1]:
        first_frame = rng.integers(low=0, high=x.shape[1] - utt_length)
        x = x[:, first_frame : first_frame + utt_length]
        logging.info(
            "extract-random-utt %s of length=%d first-frame=%d",
            key,
            x.shape[1],
            first_frame,
        )
    return x


@torch.no_grad()
def infer_qvectors(
    dataset_path: Optional[str],
    recordings_file: Optional[str],
    segments_file: Optional[str],
    vad_name: Optional[str],
    qvector_path: Optional[str],
    qmatrix_path: Optional[str],
    logits_path: Optional[str],
    vad_file: Optional[str],
    model_path: str,
    random_utt_length: bool,
    min_utt_length: float,
    max_utt_length: float,
    aug_cfg: Optional[str],
    num_augs: int,
    aug_info_path: Optional[str],
    use_gpu: bool,
    max_batch_duration: Optional[float],
    override_chunk_duration: Optional[float],
    **kwargs: Any,
) -> None:
    """Infer q-vectors (plus optional q-matrices/logits) from audio inputs.

    Args:
        dataset_path: HyperDataset manifest when reading recordings/segments jointly.
        recordings_file: Kaldi-style recordings specifier when not using a dataset.
        segments_file: Optional segments specifier (only valid without ``dataset_path``).
        vad_name: Named VAD entry inside ``dataset_path`` to select automatic VAD.
        qvector_path: Output specifier for q-vectors, or ``None`` to skip.
        qmatrix_path: Output specifier for query matrices, or ``None`` to skip.
        logits_path: Output specifier for logits, or ``None`` to skip.
        vad_file: Standalone VAD specifier (ignored when ``vad_name`` is provided).
        model_path: Checkpoint for the q-vector model.
        random_utt_length: Whether to crop a random chunk per utterance.
        min_utt_length: Minimum chunk duration in seconds when random cropping.
        max_utt_length: Maximum chunk duration in seconds when random cropping.
        aug_cfg: JSON/CFG describing augmentations applied per utterance.
        num_augs: Number of augmentations generated per utterance.
        aug_info_path: Optional CSV path storing augmentation metadata.
        use_gpu: Whether to run inference on GPU.
        max_batch_duration: Maximum audio duration (seconds) per batch chunk.
        override_chunk_duration: Force chunk duration (seconds) during inference.
        **kwargs: Additional arguments forwarded to SequentialAudioReader.
    """
    rng = np.random.default_rng(seed=1123581321 + kwargs.get("part_idx", 0))
    torch.backends.cudnn.benchmark = False
    device = init_device(use_gpu)
    model = load_model(model_path, device)
    target_sample_frequency = model.sample_frequency
    resampler = ResamplerToTargetFreq(target_sample_frequency)

    if not any([qvector_path, qmatrix_path, logits_path]):
        raise ValueError(
            "At least one of --qvector-path, --qmatrix-path, or --logits-path must be provided"
        )

    return_head_output = logits_path is not None

    qmatrix_shape = model.qmatrix_shape
    logits_dim = None
    if logits_path is not None:
        logits_dim = model.num_classes
        if logits_dim is None:
            raise ValueError(
                "logits output requested but the model head does not expose num_classes"
            )

    if aug_cfg is not None:
        augmenter = SpeechAugment.create(aug_cfg, rng=rng)
        aug_df = []
    else:
        augmenter = None
        aug_df = None
        num_augs = 1

    metadata_columns = ["speech_duration"]

    ar_args = AR.filter_args(**kwargs)
    writer_specs = {
        "qvector": qvector_path,
        "qmatrix": qmatrix_path,
        "logits": logits_path,
    }
    with ExitStack() as stack:
        writers = {}
        for name, spec in writer_specs.items():
            if spec is None:
                continue
            logging.info("opening %s output stream: %s", name, spec)
            writers[name] = stack.enter_context(
                DWF.create(spec, metadata_columns=metadata_columns)
            )

        assert (recordings_file is None) != (
            dataset_path is None
        ), "Provide either recordings_file or dataset_path (but not both)"

        if vad_name is not None:
            assert (
                dataset_path is not None
            ), "When vad_name is provided, dataset_path must also be provided"
            dataset = HyperDataset.load(dataset_path)
            assert (
                vad_name in dataset.vad_keys()
            ), f"VAD name {vad_name} not found in dataset"
            vad_file = dataset._vad_paths[vad_name]

        logging.info(f"opening input stream: {recordings_file} with args={ar_args}")
        with AR(
            dataset=dataset_path,
            recordings=recordings_file,
            segments=segments_file,
            **ar_args,
        ) as reader:
            if vad_file is not None:
                logging.info("opening VAD stream: %s", vad_file)
                v_reader = VRF.create(vad_file)

            while not reader.eof():
                t1 = time.time()
                key, x0, fs = reader.read(1)
                if len(key) == 0:
                    break

                x0 = x0[0]
                key0 = key[0]
                fs = fs[0]
                t2 = time.time()
                if fs != target_sample_frequency:
                    x0, fs = resampler(x0, fs)

                logging.info("processing utt %s", key0)
                for aug_id in range(num_augs):
                    metadata = {}
                    t3 = time.time()
                    key, x = augment(key0, x0, augmenter, aug_df, aug_id)
                    t4 = time.time()
                    x = torch.tensor(
                        x[None, :],
                        dtype=torch.get_default_dtype(),
                    )
                    t5 = time.time()
                    tot_samples = x.shape[1]
                    if vad_file is not None:
                        vad = v_reader.read(key0)[0]
                        vad = torch.tensor(vad[None, None, :], dtype=torch.float)
                        vad = torch.nn.functional.interpolate(
                            vad, size=x.size(-1), mode="nearest"
                        ).bool()[0, 0]
                        x = x[:, vad]
                    t6 = time.time()

                    if random_utt_length:
                        x = select_random_chunk(
                            key,
                            x,
                            target_sample_frequency,
                            min_utt_length,
                            max_utt_length,
                            rng,
                        )
                    t7 = time.time()

                    speech_duration = x.shape[1] / target_sample_frequency
                    metadata["speech_duration"] = speech_duration

                    qmatrix = None
                    logits = None
                    if x.shape[1] == 0:
                        logging.warning(
                            "utt %s contains no speech samples after VAD",
                            key,
                        )
                        qvector = np.zeros((model.qvector_dim,), dtype=float_cpu())
                        if qmatrix_shape is not None:
                            qmatrix = np.zeros(qmatrix_shape, dtype=float_cpu())
                        if logits_dim is not None:
                            logits = np.zeros((logits_dim,), dtype=float_cpu())
                        infer_time = 0.0
                    else:
                        audio_lengths = torch.tensor(
                            (x.shape[1],),
                            dtype=torch.long,
                        )
                        infer_start = time.time()
                        qvector_output = model.infer(
                            x,
                            audio_lengths=audio_lengths,
                            max_batch_duration=max_batch_duration,
                            override_chunk_duration=override_chunk_duration,
                            return_head_output=return_head_output,
                        )
                        infer_end = time.time()
                        infer_time = infer_end - infer_start
                        qvector = qvector_output.qvector.detach().cpu().numpy()[0]
                        if qmatrix_shape is not None:
                            qmatrix = qvector_output.qmatrix.detach().cpu().numpy()[0]
                        if logits_dim is not None:
                            head_output = qvector_output.head_output
                            if head_output is None:
                                raise RuntimeError(
                                    "Model did not return head output despite logits being requested"
                                )
                            logits_tensor = getattr(head_output, "logits", None)
                            if logits_tensor is None:
                                raise ValueError(
                                    "Head output does not expose logits tensor"
                                )
                            logits = logits_tensor.detach().cpu().numpy()[0]

                    t8 = time.time()
                    if "qvector" in writers:
                        writers["qvector"].write([key], [qvector], metadata=metadata)
                    if "qmatrix" in writers:
                        writers["qmatrix"].write([key], [qmatrix], metadata=metadata)
                    if "logits" in writers:
                        writers["logits"].write([key], [logits], metadata=metadata)
                    t9 = time.time()
                    read_time = t2 - t1
                    tot_time = read_time + t9 - t3
                    logging.info(
                        (
                            "utt %s total-time=%.3f read-time=%.3f "
                            "aug-time=%.3f tensor-time=%.3f "
                            "vad-time=%.3f chunk-time=%.3f "
                            "infer-time=%.3f write-time=%.3f "
                            "rt-factor=%.2f"
                        ),
                        key,
                        tot_time,
                        read_time,
                        t4 - t3,
                        t5 - t4,
                        t6 - t5,
                        t7 - t6,
                        infer_time,
                        t9 - t8,
                        0.0 if speech_duration == 0 else speech_duration / tot_time,
                    )

    if aug_info_path is not None and aug_df is not None:
        aug_df = pd.concat(aug_df, ignore_index=True)
        aug_df.to_csv(aug_info_path, index=False, na_rep="n/a")


def main() -> None:
    """Run the CLI entry point."""
    parser = ArgumentParser(description="Infer q-vectors from audio recordings")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="HyperDataset file describing recordings, segments, and optional VAD tables",
    )
    parser.add_argument(
        "--recordings-file",
        default=None,
        help="Kaldi-style recording specifier (scp/ark) used when no dataset is provided",
    )
    parser.add_argument(
        "--segments-file",
        default=None,
        help="Optional Kaldi segments specifier when --recordings-file is used",
    )
    parser.add_argument(
        "--vad-file",
        default=None,
        help="Binary/table VAD specifier used when no embedded dataset VAD is selected",
    )
    parser.add_argument(
        "--vad-name",
        default=None,
        help="Name of the VAD entry stored within --dataset-path to use during decoding",
    )
    AR.add_class_args(parser)

    parser.add_argument("--aug-cfg", default=None)
    parser.add_argument("--aug-info-path", default=None)
    parser.add_argument(
        "--num-augs", default=1, type=int, help="number of augmentations per utterance"
    )

    parser.add_argument("--model-path", required=True)

    parser.add_argument(
        "--random-utt-length",
        default=False,
        action="store_true",
        help="calculates q-vector from a random chunk",
    )
    parser.add_argument(
        "--min-utt-length",
        type=float,
        default=5,
        help=("minimum utterance length in secs when using random utt length"),
    )
    parser.add_argument(
        "--max-utt-length",
        type=float,
        default=120,
        help=("maximum utterance length in secs when using random utt length"),
    )
    parser.add_argument(
        "--qvector-path",
        default=None,
        help="output specifier for qvectors (e.g., ark:filepath or h5:filepath)",
    )
    parser.add_argument(
        "--qmatrix-path",
        default=None,
        help="output specifier for q-matrices per utterance",
    )
    parser.add_argument(
        "--logits-path",
        default=None,
        help="output specifier for logits (requires a classification head)",
    )
    parser.add_argument(
        "--max-batch-duration",
        type=float,
        default=None,
        help="max total audio duration per inference batch in seconds",
    )
    parser.add_argument(
        "--override-chunk-duration",
        type=float,
        default=None,
        help=(
            "override internal chunk duration (in seconds) used by the q-vector model"
        ),
    )
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="infer qvectors in gpu"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    if (
        args.qvector_path is None
        and args.qmatrix_path is None
        and args.logits_path is None
    ):
        parser.error(
            "At least one of --qvector-path, --qmatrix-path, or --logits-path must be provided"
        )
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    infer_qvectors(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
