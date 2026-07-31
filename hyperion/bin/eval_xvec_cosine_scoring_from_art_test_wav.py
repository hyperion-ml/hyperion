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
import pandas as pd
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import AudioWriter as AW
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.classifiers import BinaryLogisticRegression as LR
from hyperion.torch import HyperTorchModel
from hyperion.torch.adv_attacks.art_attack_factory import (
    ARTAttackFactory as AttackFactory,
)
from hyperion.torch.layers import LinBinCalibrator as Calibrator
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.torch.utils.misc import compute_stats_adv_attack, l2_norm
from hyperion.utils import TrialKey, TrialNdx, TrialScores, Utt2Info
from hyperion.utils.list_utils import ismember
from hyperion.utils.misc import PathLike


def init_device(use_gpu: bool) -> torch.device:
    """Initialize runtime device for evaluation.

    Args:
        use_gpu: If ``True``, request a GPU device.
    """
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_feats(**kwargs: Any) -> AF:
    """Initialize waveform feature extractor from parsed configuration.

    Args:
        **kwargs: Parsed argument dictionary containing ``feats`` config.
    """
    feat_args = AF.filter_args(**kwargs["feats"])
    logging.info("feat args={}".format(feat_args))
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=False, **feat_args)
    logging.info("feat-extractor={}".format(feat_extractor))
    feat_extractor.eval()
    return feat_extractor


def load_model(model_path: PathLike) -> nn.Module:
    """Load x-vector model checkpoint.

    Args:
        model_path: Path to a serialized torch model checkpoint.
    """
    logging.info("loading model {}".format(model_path))
    model = HyperTorchModel.auto_load(model_path)
    logging.info("xvector-model={}".format(model))
    model.eval()
    return model


def load_calibrator(cal_file: PathLike) -> Calibrator:
    """Load logistic calibration parameters and convert to torch module.

    Args:
        cal_file: Path to calibration model file.
    """
    logging.info("loading calibration params {}".format(cal_file))
    lr = LR.load(cal_file)
    calibrator = Calibrator(lr.A[0, 0], lr.b[0])
    calibrator.eval()
    return calibrator


def read_data(
    v_file: PathLike,
    key_file: PathLike,
    enroll_file: PathLike,
    seg_part_idx: int,
    num_seg_parts: int,
) -> Tuple[TrialKey, np.ndarray]:
    """Load trial key plus enrollment embeddings for evaluation.

    Args:
        v_file: Input enrollment embedding archive/specifier.
        key_file: Trial key file defining target/non-target trials.
        enroll_file: Enrollment mapping file from model ids to segment ids.
        seg_part_idx: Current test split index (1-based).
        num_seg_parts: Number of test splits.
    """
    r = DRF.create(v_file)
    enroll = Utt2Info.load(enroll_file)
    key = TrialKey.load(key_file)
    if num_seg_parts > 1:
        key = key.split(1, 1, seg_part_idx, num_seg_parts)

    x_e = r.read(enroll.key, squeeze=True)
    f, idx = ismember(key.model_set, enroll.info)
    assert np.all(f)
    x_e = x_e[idx]
    return key, x_e


class MyModel(nn.Module):
    """Wrapper model used to score enroll/test embeddings under ART attacks."""

    def __init__(
        self,
        feat_extractor: AF,
        xvector_model: nn.Module,
        embed_layer: Optional[int] = None,
        calibrator: Optional[nn.Module] = None,
        threshold: float = 0.0,
    ) -> None:
        """Initialize the scoring model.

        Args:
            feat_extractor: Front-end feature extractor operating on waveform input.
            xvector_model: X-vector model used to extract embeddings.
            embed_layer: Optional classifier layer index used for embedding extraction.
            calibrator: Optional score calibrator applied to cosine scores.
            threshold: Score threshold used to build the non-target logit.
        """
        super().__init__()
        self.feat_extractor = feat_extractor
        self.xvector_model = xvector_model
        self.x_e = None
        self.vad_t = None
        self.embed_layer = embed_layer
        self.calibrator = calibrator
        self.threshold = threshold

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """Compute target/non-target logits for ART attack/evaluation.

        Args:
            s_t: Test waveform tensor.
        """
        if s_t.dim() == 4:
            # this is for attacks that only work in 4D inputs
            s_t = s_t[0, 0]

        f_t = s_t
        f_t, _ = self.feat_extractor(s_t)
        if self.vad_t is not None:
            n_vad_frames = len(self.vad_t)
            n_feat_frames = f_t.shape[1]
            if n_vad_frames > n_feat_frames:
                self.vad_t = self.vad_t[:n_feat_frames]
            elif n_vad_frames < n_feat_frames:
                f_t = f_t[:, :n_vad_frames]

            f_t = f_t[:, self.vad_t]

        f_t = f_t.transpose(1, 2).contiguous()
        x_t = self.xvector_model.extract_embed(f_t, embed_layer=self.embed_layer)
        if self.x_e is None:
            # this is for auto-pgd, when it runs a dummy evaluation
            self.x_e = x_t

        x_t = l2_norm(x_t)
        x_e = l2_norm(self.x_e)
        tar_score = torch.sum(x_e * x_t, dim=-1, keepdim=True)
        if self.calibrator is not None:
            score = self.calibrator(tar_score)

        non_score = self.threshold + 0 * tar_score
        score = torch.cat((non_score, tar_score), dim=-1)  # .unsqueeze(0)
        return score


def eval_cosine_scoring(
    v_file: PathLike,
    key_file: PathLike,
    enroll_file: PathLike,
    test_wav_file: PathLike,
    vad_spec: Optional[PathLike],
    vad_path_prefix: Optional[PathLike],
    model_path: PathLike,
    embed_layer: Optional[int],
    score_file: PathLike,
    stats_file: Optional[PathLike],
    cal_file: Optional[PathLike],
    threshold: float,
    save_adv_wav: bool,
    save_adv_wav_path: Optional[PathLike],
    max_test_length: Optional[float],
    use_gpu: bool,
    seg_part_idx: int,
    num_seg_parts: int,
    **kwargs: Any,
) -> None:
    """Evaluate adversarial cosine scoring on test waveforms using ART attacks.

    Args:
        v_file: Enrollment embedding archive/specifier.
        key_file: Trial key file defining target/non-target pairs.
        enroll_file: Enrollment mapping file.
        test_wav_file: Test waveform recordings specifier.
        vad_spec: Optional VAD specifier for selecting speech frames.
        vad_path_prefix: Optional path prefix applied to VAD entries.
        model_path: X-vector model checkpoint path.
        embed_layer: Optional embedding layer index to extract.
        score_file: Output file for trial scores.
        stats_file: Output CSV path for adversarial perturbation statistics.
        cal_file: Optional calibration model file.
        threshold: Decision threshold for calibrated scores.
        save_adv_wav: Whether to write adversarial examples to disk.
        save_adv_wav_path: Output directory for saved adversarial waveforms.
        max_test_length: Optional max test duration in seconds.
        use_gpu: Whether to run inference/attacks on GPU.
        seg_part_idx: Test split index (1-based).
        num_seg_parts: Total number of test splits.
        **kwargs: Additional parsed args, including attack/reader/feature settings.
    """
    device_type = "gpu" if use_gpu else "cpu"
    device = init_device(use_gpu)
    feat_extractor = init_feats(**kwargs)
    xvector_model = load_model(model_path)

    calibrator = None
    if cal_file is not None:
        calibrator = load_calibrator(cal_file)

    model = MyModel(
        feat_extractor, xvector_model, embed_layer, calibrator, threshold=threshold
    )
    model.to(device)
    model.eval()

    tar = np.asarray([1], dtype=int)
    non = np.asarray([0], dtype=int)

    logging.info("loading key and enrollment x-vectors")
    key, x_e = read_data(v_file, key_file, enroll_file, seg_part_idx, num_seg_parts)
    x_e = torch.as_tensor(x_e, dtype=torch.get_default_dtype())

    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(recordings=test_wav_file, **audio_args)
    wav_scale = audio_reader.wav_scale

    if save_adv_wav:
        tar_audio_writer = AW(save_adv_wav_path + "/tar2non")
        non_audio_writer = AW(save_adv_wav_path + "/non2tar")

    attack_args = AttackFactory.filter_args(**kwargs["attack"])
    extra_args = {"eps_scale": wav_scale}
    attack_args.update(extra_args)
    logging.info("attack-args={}".format(attack_args))

    if vad_spec is not None:
        logging.info("opening VAD stream: %s" % (vad_spec))
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

    scores = np.zeros((key.num_models, key.num_tests), dtype="float32")
    attack_stats = pd.DataFrame(
        columns=[
            "modelid",
            "segmentid",
            "snr",
            "px",
            "pn",
            "x_l2",
            "x_linf",
            "n_l0",
            "n_l2",
            "n_linf",
            "num_frames",
        ]
    )

    for j in range(key.num_tests):
        t1 = time.time()
        logging.info("scoring test utt %s", key.seg_set[j])
        s, fs = audio_reader.read([key.seg_set[j]])
        s = s[0]
        fs = fs[0]

        if max_test_length is not None:
            max_samples = int(fs * max_test_length)
            if len(s) > max_samples:
                s = s[:max_samples]

        s = s[None, :].astype("float32", copy=False)
        s_tensor = torch.as_tensor(s, dtype=torch.get_default_dtype()).to(device)

        if vad_spec is not None:
            vad = v_reader.read([key.seg_set[j]])[0]
            tot_frames = len(vad)
            speech_frames = np.sum(vad)
            vad = torch.tensor(vad, dtype=torch.bool).to(device)
            model.vad_t = vad
            logging.info(
                "utt %s detected %d/%d (%.2f %%) speech frames",
                key.seg_set[j],
                speech_frames,
                tot_frames,
                speech_frames / tot_frames * 100,
            )

        t2 = time.time()

        trial_time = 0
        num_trials = 0
        model_art = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=(s.shape[1],),
            nb_classes=2,
            clip_values=(-wav_scale, wav_scale),
            device_type=device_type,
        )

        attack_args["num_samples"] = s.shape[-1]
        attack = AttackFactory.create(model_art, **attack_args)
        # s = s[None, None, :, :]
        for i in range(key.num_models):
            if key.tar[i, j] or key.non[i, j]:
                t3 = time.time()
                model.x_e = x_e[i : i + 1].to(device)
                if key.tar[i, j]:
                    if attack.targeted:
                        t = non
                    else:
                        t = tar
                else:
                    if attack.targeted:
                        t = tar
                    else:
                        t = non

                s_adv = attack.generate(s, t)
                # s_adv = s_adv[0, 0]
                s_adv = torch.from_numpy(s_adv).to(device)
                with torch.no_grad():
                    scores[i, j] = model(s_adv).cpu().numpy()[0, 1]

                t4 = time.time()
                trial_time += t4 - t3
                num_trials += 1

                s_adv = s_adv.detach()
                stats_ij = compute_stats_adv_attack(s_tensor, s_adv)
                stats_ij = [stat.detach().cpu().numpy()[0] for stat in stats_ij]
                attack_stats = attack_stats.append(
                    {
                        "modelid": key.model_set[i],
                        "segmentid": key.seg_set[j],
                        "snr": stats_ij[0],
                        "px": stats_ij[1],
                        "pn": stats_ij[2],
                        "x_l2": stats_ij[3],
                        "x_linf": stats_ij[4],
                        "n_l0": stats_ij[5],
                        "n_l2": stats_ij[6],
                        "n_linf": stats_ij[7],
                        "num_samples": s.shape[-1],
                    },
                    ignore_index=True,
                )

                # logging.info('min-max %f %f %f %f' % (torch.min(s), torch.max(s), torch.min(s_adv-s), torch.max(s_adv-s)))
                if save_adv_wav:
                    s_adv = s_adv.cpu().numpy()[0]
                    trial_name = "%s-%s" % (key.model_set[i], key.seg_set[j])
                    if key.tar[i, j] and scores[i, j] < threshold:
                        tar_audio_writer.write(trial_name, s_adv, fs)
                    elif key.non[i, j] and scores[i, j] > threshold:
                        non_audio_writer.write(trial_name, s_adv, fs)

        del attack
        del model_art
        trial_time /= num_trials
        t7 = time.time()
        logging.info(
            (
                "utt %s total-time=%.3f read-time=%.3f trial-time=%.3f n_trials=%d "
                "rt-factor=%.5f"
            ),
            key.seg_set[j],
            t7 - t1,
            t2 - t1,
            trial_time,
            num_trials,
            (t7 - t1) / (num_trials * s.shape[1] / fs),
        )

    if num_seg_parts > 1:
        score_file = "%s-%03d-%03d" % (score_file, 1, seg_part_idx)
        stats_file = "%s-%03d-%03d" % (stats_file, 1, seg_part_idx)
    logging.info("saving scores to %s", score_file)
    s = TrialScores(
        key.model_set, key.seg_set, scores, score_mask=np.logical_or(key.tar, key.non)
    )
    s.save_txt(score_file)

    logging.info("saving stats to %s", stats_file)
    attack_stats.to_csv(stats_file)


def main() -> None:
    """Parse CLI arguments and run ART-based adversarial cosine-scoring evaluation.

    Args:
        None.
    """
    parser = ArgumentParser(
        description=(
            "Eval cosine-scoring given enroll x-vector "
            "and adversarial test wave from ART"
        )
    )

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--v-file",
        required=True,
        help="enrollment x-vector archive/specifier",
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="trial key file containing target/non-target labels",
    )
    parser.add_argument(
        "--enroll-file",
        required=True,
        help="enrollment map file linking models to enrollment segment ids",
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

    AttackFactory.add_class_args(parser, prefix="attack")

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
        dest="score_file",
        required=True,
        help="output file path for trial scores",
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
        "--save-adv-wav",
        default=False,
        action="store_true",
        help="save adversarial signals to disk",
    )
    parser.add_argument(
        "--save-adv-wav-path",
        default=None,
        help="output directory for saved adversarial waveforms",
    )

    parser.add_argument(
        "--stats-file",
        default=None,
        help="output CSV path for adversarial attack statistics",
    )

    parser.add_argument(
        "--cal-file",
        default=None,
        help="optional logistic calibration model file",
    )
    parser.add_argument("--threshold", default=0, type=float, help="decision threshold")
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
