#!/usr/bin/env python
"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import sys
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.classifiers import BinaryLogisticRegression as LR
from hyperion.torch import HyperTorchModel
from hyperion.torch.layers import LinBinCalibrator as Calibrator
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.torch.utils.misc import l2_norm
from hyperion.utils import TrialKey, TrialNdx, TrialScores, Utt2Info
from hyperion.utils.list_utils import ismember
from hyperion.utils.misc import PathLike


def init_device(use_gpu: bool) -> torch.device:
    """Initialize runtime device for evaluation.

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


def load_model(model_path: PathLike, device: torch.device) -> nn.Module:
    """Load x-vector model checkpoint.

    Args:
        model_path: Path to serialized torch model checkpoint.
        device: Torch device where the model is loaded.
    """
    logging.info("loading model {}".format(model_path))
    model = HyperTorchModel.auto_load(model_path)
    logging.info("xvector-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def load_calibrator(cal_file: PathLike, device: torch.device) -> Calibrator:
    """Load score calibrator from logistic-regression parameters.

    Args:
        cal_file: Path to logistic-regression calibration model.
        device: Torch device where the calibrator runs.
    """
    logging.info("loading calibration params {}".format(cal_file))
    lr = LR.load(cal_file)
    calibrator = Calibrator(lr.A[0, 0], lr.b[0])
    calibrator.to(device)
    calibrator.eval()
    return calibrator


def read_data(
    v_file: PathLike,
    ndx_file: PathLike,
    enroll_file: PathLike,
    seg_part_idx: int,
    num_seg_parts: int,
) -> Tuple[TrialNdx, np.ndarray]:
    """Load trial index and enrollment embeddings.

    Args:
        v_file: Input enrollment embedding archive/specifier.
        ndx_file: Trial index/key file defining enroll-test trial mask.
        enroll_file: Enrollment map file linking model ids and segments.
        seg_part_idx: Test split index (1-based).
        num_seg_parts: Number of test splits.
    """
    r = DRF.create(v_file)
    enroll = Utt2Info.load(enroll_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file).to_ndx()

    if num_seg_parts > 1:
        ndx = ndx.split(1, 1, seg_part_idx, num_seg_parts)

    x_e = r.read(enroll.key, squeeze=True)

    f, idx = ismember(ndx.model_set, enroll.info)

    assert np.all(f)
    x_e = x_e[idx]

    return ndx, x_e


def eval_cosine_scoring(
    v_file: PathLike,
    ndx_file: PathLike,
    enroll_file: PathLike,
    test_wav_file: PathLike,
    vad_spec: Optional[PathLike],
    vad_path_prefix: Optional[PathLike],
    model_path: PathLike,
    embed_layer: Optional[int],
    score_file: PathLike,
    cal_file: Optional[PathLike],
    max_test_length: Optional[float],
    use_gpu: bool,
    seg_part_idx: int,
    num_seg_parts: int,
    **kwargs: Any,
) -> None:
    """Evaluate cosine-scoring using enrollment x-vectors and test waveforms.

    Args:
        v_file: Enrollment embedding archive/specifier.
        ndx_file: Trial index/key file defining evaluation trials.
        enroll_file: Enrollment mapping file.
        test_wav_file: Test waveform recordings specifier.
        vad_spec: Optional VAD specifier for speech-frame selection.
        vad_path_prefix: Optional path prefix applied to VAD entries.
        model_path: X-vector model checkpoint path.
        embed_layer: Optional classifier layer used to extract embeddings.
        score_file: Output path for trial scores.
        cal_file: Optional score calibration model file.
        max_test_length: Optional max test duration in seconds.
        use_gpu: Whether to run evaluation on GPU.
        seg_part_idx: Test split index (1-based) when parallelized.
        num_seg_parts: Total number of test splits.
        **kwargs: Additional parsed args, including reader and feature settings.
    """
    device = init_device(use_gpu)
    feat_extractor = init_feats(device, **kwargs)
    model = load_model(model_path, device)

    calibrator = None
    if cal_file is not None:
        calibrator = load_calibrator(cal_file, device)

    logging.info("loading ndx and enrollment x-vectors")
    ndx, y_e = read_data(v_file, ndx_file, enroll_file, seg_part_idx, num_seg_parts)

    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(recordings=test_wav_file, **audio_args)

    if vad_spec is not None:
        logging.info("opening VAD stream: %s", vad_spec)
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

    scores = np.zeros((ndx.num_models, ndx.num_tests), dtype="float32")
    with torch.no_grad():
        for j in range(ndx.num_tests):
            t1 = time.time()
            logging.info("scoring test utt %s", ndx.seg_set[j])
            s, fs = audio_reader.read([ndx.seg_set[j]])
            s = s[0]
            fs = fs[0]

            if max_test_length is not None:
                max_samples = int(fs * max_test_length)
                if len(s) > max_samples:
                    s = s[:max_samples]

            t2 = time.time()
            s = torch.as_tensor(s[None, :], dtype=torch.get_default_dtype()).to(device)
            x_t, _ = feat_extractor(s)
            t4 = time.time()
            tot_frames = x_t.shape[1]
            if vad_spec is not None:
                vad = v_reader.read([ndx.seg_set[j]], num_frames=x_t.shape[1])[0]
                vad = torch.tensor(vad, dtype=torch.bool).to(device)
                x_t = x_t[:, vad]
                logging.info(
                    "utt %s detected %d/%d (%.2f %%) speech frames",
                    ndx.seg_set[j],
                    x_t.shape[1],
                    tot_frames,
                    x_t.shape[1] / tot_frames * 100,
                )

            t5 = time.time()
            x_t = x_t.transpose(1, 2).contiguous()
            y_t = model.extract_embed(x_t, embed_layer=embed_layer)
            y_t = l2_norm(y_t)
            t6 = time.time()

            for i in range(ndx.num_models):
                if ndx.trial_mask[i, j]:
                    y_e_i = torch.as_tensor(
                        y_e[i : i + 1], dtype=torch.get_default_dtype()
                    ).to(device)
                    y_e_i = l2_norm(y_e_i)
                    scores_ij = torch.sum(y_e_i * y_t, dim=-1)
                    if calibrator is None:
                        scores[i, j] = scores_ij
                    else:
                        scores[i, j] = calibrator(scores_ij)

            t7 = time.time()
            num_trials = np.sum(ndx.trial_mask[:, j])
            trial_time = (t7 - t6) / num_trials
            logging.info(
                (
                    "utt %s total-time=%.3f read-time=%.3f feat-time=%.3f "
                    "vad-time=%.3f embed-time=%.3f trial-time=%.3f n_trials=%d "
                    "rt-factor=%.2f"
                ),
                ndx.seg_set[j],
                t7 - t1,
                t2 - t1,
                t4 - t2,
                t5 - t4,
                t6 - t5,
                trial_time,
                num_trials,
                (t7 - t1) / (num_trials * s.shape[1] / fs),
            )

    if num_seg_parts > 1:
        score_file = "%s-%03d-%03d" % (score_file, 1, seg_part_idx)
    logging.info("saving scores to %s", score_file)
    s = TrialScores(ndx.model_set, ndx.seg_set, scores, score_mask=ndx.trial_mask)
    s.save_txt(score_file)


def main() -> None:
    """Parse CLI arguments and run cosine-scoring evaluation.

    Args:
        None.
    """
    parser = ArgumentParser(
        description="Eval cosine-scoring given enroll x-vector and test wave"
    )

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--v-file",
        required=True,
        help="enrollment x-vector archive/specifier",
    )
    parser.add_argument(
        "--ndx-file",
        default=None,
        help="trial index/key file defining evaluation trials",
    )
    parser.add_argument(
        "--enroll-file",
        required=True,
        help="enrollment map file linking models to segment ids",
    )
    parser.add_argument(
        "--test-wav-file",
        required=True,
        help="test waveform recordings specifier",
    )

    AR.add_class_args(parser)
    AF.add_class_args(parser, prefix="feats")

    parser.add_argument(
        "--vad",
        dest="vad_spec",
        default=None,
        help="optional VAD specifier for frame selection",
    )
    parser.add_argument(
        "--vad-path-prefix",
        default=None,
        help="optional prefix for VAD scp file paths",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="x-vector model checkpoint path",
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
        "--use-gpu", default=False, action="store_true", help="run evaluation on GPU"
    )

    parser.add_argument(
        "--seg-part-idx",
        default=1,
        type=int,
        help="test split index (1-based) when evaluating in parallel",
    )
    parser.add_argument(
        "--num-seg-parts",
        default=1,
        type=int,
        help=(
            "number of parts in which we divide the test list "
            "to run evaluation in parallel"
        ),
    )

    parser.add_argument(
        "--score-file",
        required=True,
        help="output file path for trial scores",
    )
    parser.add_argument(
        "--cal-file",
        default=None,
        help="optional logistic calibration model file",
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
    parser.add_argument(
        "--max-test-length",
        default=None,
        type=float,
        help=(
            "maximum length (secs) for the test side, "
            "this is to avoid GPU memory errors"
        ),
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_cosine_scoring(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
