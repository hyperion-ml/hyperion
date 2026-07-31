"""
Copyright 2019 Jesus Villalba (Johns Hopkins University)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.augment import SpeechAugment
from hyperion.np.preprocessing import ResamplerToTargetFreq

# from hyperion.torch import TorchModelLoader as TML
from hyperion.torch import HyperTorchModel
from hyperion.torch.utils import open_device
from hyperion.utils import HyperDataset, Utt2Info


def init_device(use_gpu: bool) -> torch.device:
    """Initialise device for inference."""
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus=%d", num_gpus)
    device = open_device(num_gpus=num_gpus)
    return device


def load_model(model_path: str, device: torch.device) -> HyperTorchModel:
    """Load serialized model onto ``device``."""
    logging.info("loading model %s", model_path)
    model = HyperTorchModel.auto_load(model_path)
    logging.info(f"xvector-model={model}")
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
    """Randomly crop the utterance between ``min``/``max`` seconds."""
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


def eval_xvector_logits(
    dataset_path: Optional[str],
    recordings_file: Optional[str],
    segments_file: Optional[str],
    output_spec: Optional[str],
    logits_path: Optional[str],
    vad_spec: Optional[str],
    vad_file: Optional[str],
    vad_name: Optional[str],
    write_speech_dur: Optional[str],
    vad_path_prefix: Optional[str],
    model_path: str,
    chunk_length: float,
    random_utt_length: bool,
    min_utt_length: float,
    max_utt_length: float,
    aug_cfg: Optional[str],
    num_augs: int,
    aug_info_path: Optional[str],
    use_gpu: bool,
    **kwargs: Any,
) -> None:
    """Compute logits for each utterance using wav2xvector models."""
    rng = np.random.default_rng(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    model = load_model(model_path, device)
    resampler = ResamplerToTargetFreq(model.sample_frequency)

    if dataset_path is None and recordings_file is None:
        raise ValueError("Provide either --dataset-path or --recordings-file")
    if dataset_path is not None and recordings_file is not None:
        raise ValueError("--dataset-path and --recordings-file are mutually exclusive")
    if dataset_path is not None and segments_file is not None:
        raise ValueError("--segments-file cannot be used with --dataset-path")

    if output_spec is None and logits_path is None:
        raise ValueError(
            "At least one of --logits-path or --output-spec must be provided"
        )

    actual_vad_spec = vad_file if vad_file is not None else vad_spec
    if actual_vad_spec is None and vad_name is not None:
        if dataset_path is None:
            raise ValueError("--vad-name requires --dataset-path")
        dataset = HyperDataset.load(dataset_path)
        if vad_name not in dataset.vad_keys():
            raise ValueError(f"VAD name {vad_name} not found in dataset")
        actual_vad_spec = dataset._vad_paths[vad_name]

    if write_speech_dur is not None:
        keys = []
        info = []

    if aug_cfg is not None:
        augmenter = SpeechAugment.create(aug_cfg, rng=rng)
        aug_df = []
    else:
        augmenter = None
        aug_df = None
        num_augs = 1

    metadata_columns = ["speech_duration"]
    writer_spec = logits_path if logits_path is not None else output_spec

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output stream: %s with args=%s", writer_spec, str(ar_args))
    with DWF.create(writer_spec, metadata_columns=metadata_columns) as writer:
        input_stream = dataset_path if dataset_path is not None else recordings_file
        logging.info(f"opening input stream: {input_stream} with args={ar_args}")
        with AR(
            dataset=dataset_path,
            recordings=recordings_file,
            segments=segments_file,
            **ar_args,
        ) as reader:
            if actual_vad_spec is not None:
                logging.info("opening VAD stream: %s", actual_vad_spec)
                v_reader = VRF.create(actual_vad_spec, path_prefix=vad_path_prefix)

            while not reader.eof():
                t1 = time.time()
                key, x0, fs = reader.read(1)
                if len(key) == 0:
                    break

                x0 = x0[0]
                key0 = key[0]
                fs = fs[0]
                t2 = time.time()
                if fs != model.sample_frequency:
                    x0, fs = resampler(x0, fs)

                logging.info("processing utt %s", key0)
                for aug_id in range(num_augs):
                    metadata = {}
                    t3 = time.time()
                    key, x = augment(key0, x0, augmenter, aug_df, aug_id)
                    t4 = time.time()
                    with torch.no_grad():
                        x = torch.tensor(
                            x[None, :], dtype=torch.get_default_dtype()
                        ).to(device)
                        t5 = time.time()
                        tot_samples = x.shape[1]
                        if actual_vad_spec is not None:
                            vad = v_reader.read(key0)[0]
                            vad = torch.tensor(
                                vad[None, None, :], dtype=torch.float
                            ).to(device)
                            vad = torch.nn.functional.interpolate(
                                vad, size=x.size(-1), mode="nearest"
                            ).bool()[0, 0]
                            x = x[:, vad]

                        logging.info(
                            "utt %s detected %d/%d (%.2f %%) speech samples",
                            key,
                            x.shape[1],
                            tot_samples,
                            x.shape[1] / tot_samples * 100,
                        )

                        if random_utt_length:
                            x = select_random_chunk(
                                key, x, fs, min_utt_length, max_utt_length, rng
                            )

                        metadata["speech_duration"] = (
                            x.shape[1] / model.sample_frequency
                        )

                        t6 = time.time()
                        if x.shape[1] == 0:
                            y = np.zeros((model.num_classes,), dtype=float_cpu())
                        else:
                            y = model(x).logits.cpu().numpy()[0]

                    t7 = time.time()
                    writer.write([key], [y], metadata=metadata)
                    if write_speech_dur is not None:
                        keys.append(key)
                        info.append(str(x.shape[1] / fs))

                    t8 = time.time()
                    read_time = t2 - t1
                    tot_time = read_time + t8 - t3
                    logging.info(
                        (
                            "utt %s total-time=%.3f read-time=%.3f "
                            "aug-time=%.3f feat-time=%.3f "
                            "vad-time=%.3f embed-time=%.3f write-time=%.3f "
                            "rt-factor=%.2f"
                        ),
                        key,
                        tot_time,
                        read_time,
                        t4 - t3,
                        t5 - t4,
                        t6 - t5,
                        t7 - t6,
                        t8 - t7,
                        x.shape[1] / fs / tot_time,
                    )

    if write_speech_dur is not None:
        logging.info("writing speech duration in secs to %s", write_speech_dur)
        u2sd = Utt2Info.create(keys, info)
        u2sd.save(write_speech_dur)

    if aug_info_path is not None:
        aug_df = pd.concat(aug_df, ignore_index=True)
        aug_df.to_csv(aug_info_path, index=False, na_rep="n/a")


def main() -> None:
    """CLI entry point."""
    parser = ArgumentParser(
        description="""Extracts x-vectors from waveform computing acoustic features on the fly"""
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="HyperDataset manifest describing recordings and optional VAD tables",
    )
    parser.add_argument(
        "--recordings-file",
        default=None,
        help="Recording specifier used when no dataset is provided",
    )
    parser.add_argument(
        "--segments-file",
        default=None,
        help="Kaldi segments file (only valid when using --recordings-file)",
    )
    parser.add_argument(
        "--vad",
        dest="vad_spec",
        default=None,
        help="(Deprecated) VAD specifier kept for backward compatibility",
    )
    parser.add_argument(
        "--vad-file",
        default=None,
        help="Standalone VAD specifier that overrides --vad when provided",
    )
    parser.add_argument(
        "--vad-name",
        default=None,
        help="Name of VAD entry stored in the dataset manifest",
    )
    parser.add_argument(
        "--write-speech-dur",
        default=None,
        help="Path to store utt2dur information (seconds of detected speech)",
    )
    parser.add_argument(
        "--vad-path-prefix", default=None, help=("scp file_path prefix for vad")
    )

    AR.add_class_args(parser)

    parser.add_argument("--aug-cfg", default=None)
    parser.add_argument("--aug-info-path", default=None)
    parser.add_argument(
        "--num-augs", default=1, type=int, help="number of augmentations per utterance"
    )

    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--chunk-length",
        type=float,
        default=0,
        help=(
            "max. chunk length used in each forward pass "
            "of the x-vector encoder,"
            "if 0 the full utterance is used"
        ),
    )

    parser.add_argument(
        "--random-utt-length",
        default=False,
        action="store_true",
        help="calculates x-vector from a random chunk",
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
        "--logits-path",
        default=None,
        help="Preferred output specifier (ark,h5, etc.) for logits",
    )
    parser.add_argument(
        "--output-spec",
        default=None,
        help="Legacy output specifier; used when --logits-path is not provided",
    )
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    if args.dataset_path is None and args.recordings_file is None:
        parser.error("Provide either --dataset-path or --recordings-file")
    if args.dataset_path is not None and args.recordings_file is not None:
        parser.error("--dataset-path and --recordings-file cannot be used together")
    if args.dataset_path is not None and args.segments_file is not None:
        parser.error("--segments-file cannot be used with --dataset-path")
    if args.vad_name is not None and args.dataset_path is None:
        parser.error("--vad-name requires --dataset-path")
    if args.output_spec is None and args.logits_path is None:
        parser.error("Provide --logits-path or --output-spec")

    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_xvector_logits(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
