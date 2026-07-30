#!/usr/bin/env python
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
from hyperion.torch import HyperTorchModel
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.utils import Utt2Info
from hyperion.utils.misc import PathLike


def init_device(use_gpu: bool) -> torch.device:
    """Initialize runtime device for extraction.

    Args:
        use_gpu: If ``True``, request one GPU device.
    """
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_feats(device: torch.device, **kwargs: Any) -> AF:
    """Initialize waveform feature extractor from parsed configuration.

    Args:
        device: Torch device where feature extraction runs.
        **kwargs: Parsed argument dictionary containing ``feats`` config.
    """
    feat_args = AF.filter_args(**kwargs["feats"])
    logging.info("feat args={}".format(feat_args))
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=False, **feat_args)
    logging.info("feat-extractor={}".format(feat_extractor))
    feat_extractor.eval()
    feat_extractor.to(device)
    return feat_extractor


def load_model(model_path: PathLike, device: torch.device) -> torch.nn.Module:
    """Load x-vector model checkpoint.

    Args:
        model_path: Path to serialized model checkpoint.
        device: Torch device where inference runs.
    """
    logging.info("loading model {}".format(model_path))
    model = HyperTorchModel.auto_load(model_path)
    logging.info("xvector-model={}".format(model))
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
    """Apply optional augmentation and collect augmentation metadata.

    Args:
        key0: Original utterance key.
        x0: Original waveform samples.
        augmenter: Optional speech augmenter instance.
        aug_df: Optional list accumulating augmentation metadata rows.
        aug_id: Augmentation index appended to key suffix.
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
    min_utt_length: int,
    max_utt_length: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Randomly crop the feature sequence between min and max frame lengths.

    Args:
        key: Utterance key used for logging.
        x: Feature tensor with time dimension at axis 1.
        min_utt_length: Minimum random chunk length in frames.
        max_utt_length: Maximum random chunk length in frames.
        rng: Numpy random generator.
    """
    utt_length = rng.integers(low=min_utt_length, high=max_utt_length + 1)
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


def extract_xvectors(
    recordings_file: PathLike,
    output_spec: PathLike,
    vad_spec: Optional[PathLike],
    write_num_frames_spec: Optional[PathLike],
    vad_path_prefix: Optional[PathLike],
    model_path: PathLike,
    chunk_length: int,
    embed_layer: Optional[int],
    random_utt_length: bool,
    min_utt_length: int,
    max_utt_length: int,
    aug_cfg: Optional[PathLike],
    num_augs: int,
    aug_info_path: Optional[PathLike],
    use_gpu: bool,
    **kwargs: Any,
) -> None:
    """Extract x-vectors from waveforms and write embeddings to disk.

    Args:
        recordings_file: Input recordings specifier.
        output_spec: Output writer specifier for x-vectors.
        vad_spec: Optional VAD specifier for frame selection.
        write_num_frames_spec: Optional output file for effective frame counts.
        vad_path_prefix: Optional path prefix applied to VAD entries.
        model_path: Model checkpoint path.
        chunk_length: Frames per encoder forward pass (0 means full utterance).
        embed_layer: Optional classifier layer index for embedding extraction.
        random_utt_length: Whether to extract from a random frame chunk.
        min_utt_length: Minimum random chunk length in frames.
        max_utt_length: Maximum random chunk length in frames.
        aug_cfg: Optional augmentation configuration file.
        num_augs: Number of augmentations per utterance.
        aug_info_path: Optional CSV output path for augmentation metadata.
        use_gpu: Whether to run extraction on GPU.
        **kwargs: Additional parsed args for readers/features/partitioning.
    """
    rng = np.random.default_rng(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    feat_extractor = init_feats(device, **kwargs)
    model = load_model(model_path, device)

    if write_num_frames_spec is not None:
        keys = []
        info = []

    if aug_cfg is not None:
        augmenter = SpeechAugment.create(aug_cfg, rng=rng)
        aug_df = []
    else:
        augmenter = None
        aug_df = None
        num_augs = 1

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output stream: %s", output_spec)
    with DWF.create(output_spec) as writer:
        logging.info(
            "opening input stream: {} with args={}".format(recordings_file, ar_args)
        )
        with AR(recordings=recordings_file, **ar_args) as reader:
            if vad_spec is not None:
                logging.info("opening VAD stream: %s", vad_spec)
                v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

            while not reader.eof():
                t1 = time.time()
                key, x0, fs = reader.read(1)
                if len(key) == 0:
                    break

                x0 = x0[0]
                key0 = key[0]
                t2 = time.time()

                logging.info("processing utt %s", key0)
                for aug_id in range(num_augs):
                    t3 = time.time()
                    key, x = augment(key0, x0, augmenter, aug_df, aug_id)
                    t4 = time.time()
                    with torch.no_grad():
                        x = torch.tensor(
                            x[None, :], dtype=torch.get_default_dtype()
                        ).to(device)

                        x, _ = feat_extractor(x)
                        t5 = time.time()
                        tot_frames = x.shape[1]
                        if vad_spec is not None:
                            vad = v_reader.read(key0, num_frames=tot_frames)[0]
                            vad = torch.tensor(vad, dtype=torch.bool).to(device)
                            x = x[:, vad]

                        logging.info(
                            "utt %s detected %d/%d (%.2f %%) speech frames",
                            key,
                            x.shape[1],
                            tot_frames,
                            x.shape[1] / tot_frames * 100,
                        )

                        if random_utt_length:
                            x = select_random_chunk(
                                key, x, min_utt_length, max_utt_length, rng
                            )

                        t6 = time.time()
                        if x.shape[1] == 0:
                            y = np.zeros((model.embed_dim,), dtype=float_cpu())
                        else:
                            x = x.transpose(1, 2).contiguous()
                            y = (
                                model.extract_embed(
                                    x,
                                    chunk_length=chunk_length,
                                    embed_layer=embed_layer,
                                )
                                .cpu()
                                .numpy()[0]
                            )

                    t7 = time.time()
                    writer.write([key], [y])
                    if write_num_frames_spec is not None:
                        keys.append(key)
                        info.append(str(x.shape[-1]))

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
                        x0.shape[0] / fs[0] / tot_time,
                    )

    if write_num_frames_spec is not None:
        logging.info("writing num-frames to %s", write_num_frames_spec)
        u2nf = Utt2Info.create(keys, info)
        u2nf.save(write_num_frames_spec)

    if aug_info_path is not None:
        aug_df = pd.concat(aug_df, ignore_index=True)
        aug_df.to_csv(aug_info_path, index=False, na_rep="n/a")


def main() -> None:
    """Parse CLI arguments and run x-vector extraction from waveforms.

    Args:
        None.
    """
    parser = ArgumentParser(
        description=(
            "Extracts x-vectors from waveform computing acoustic features on the fly"
        )
    )

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--recordings-file",
        required=True,
        help="input waveform recordings specifier",
    )
    parser.add_argument(
        "--vad",
        dest="vad_spec",
        default=None,
        help="optional VAD specifier for frame selection",
    )
    parser.add_argument(
        "--write-num-frames",
        dest="write_num_frames_spec",
        default=None,
        help="optional output file for effective frame counts",
    )
    parser.add_argument(
        "--vad-path-prefix",
        default=None,
        help="optional prefix for VAD scp file paths",
    )

    AR.add_class_args(parser)

    parser.add_argument(
        "--aug-cfg",
        default=None,
        help="optional speech-augmentation configuration file",
    )
    parser.add_argument(
        "--aug-info-path",
        default=None,
        help="optional CSV output path for augmentation metadata",
    )
    parser.add_argument(
        "--num-augs", default=1, type=int, help="number of augmentations per utterance"
    )

    AF.add_class_args(parser, prefix="feats")

    parser.add_argument("--model-path", required=True, help="model checkpoint path")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=0,
        help=(
            "number of frames used in each forward pass "
            "of the x-vector encoder,"
            "if 0 the full utterance is used"
        ),
    )
    parser.add_argument(
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "classifier layer used to extract embeddings; if omitted, "
            "the training-time default layer is used"
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
        type=int,
        default=500,
        help=("minimum utterance length when using random utt length"),
    )
    parser.add_argument(
        "--max-utt-length",
        type=int,
        default=12000,
        help=("maximum utterance length when using random utt length"),
    )

    parser.add_argument(
        "--output-spec",
        required=True,
        help="output writer specifier for x-vectors",
    )
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="run extraction on GPU"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=warning, 1=info, 2=debug, 3=trace)",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_xvectors(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
