#!/usr/bin/env python
"""
  Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
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
from hyperion.torch import TorchModelLoader as TML
from hyperion.torch.adv_attacks import RandomAttackFactory
from hyperion.torch.layers import LinBinCalibrator as Calibrator
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.torch.utils.misc import compute_stats_adv_attack, l2_norm
from hyperion.utils import TrialKey, TrialNdx, TrialScores, Utt2Info
from hyperion.utils.list_utils import ismember
from hyperion.utils.misc import PathLike


class MyModel(nn.Module):
    """Wrapper model for verification scoring during adversarial attacks."""

    def __init__(
        self,
        feat_extractor: AF,
        xvector_model: nn.Module,
        embed_layer: Optional[int] = None,
        calibrator: Optional[nn.Module] = None,
        sigma: float = 0.0,
    ) -> None:
        """Initialize model components used to score verification trials.

        Args:
            feat_extractor: Acoustic feature extractor.
            xvector_model: X-vector model used to produce embeddings.
            embed_layer: Optional embedding layer index for extraction.
            calibrator: Optional score calibrator applied after cosine scoring.
            sigma: Standard deviation of Gaussian input noise.
        """
        super().__init__()
        self.feat_extractor = feat_extractor
        self.xvector_model = xvector_model
        self.x_e: Optional[torch.Tensor] = None
        self.vad_t: Optional[torch.Tensor] = None
        self.embed_layer = embed_layer
        self.calibrator = calibrator
        self.sigma = sigma

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """Compute verification score for a waveform tensor.

        Args:
            s_t: Input waveform tensor of shape ``(batch, samples)``.
        """
        # print('sigma0=', self.sigma)
        if self.sigma > 0:
            s_t = s_t + self.sigma * torch.randn_like(s_t)
            # print('sigma1=', self.sigma)
        f_t = self.feat_extractor(s_t)
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
        x_t = l2_norm(x_t)
        x_e = l2_norm(self.x_e)
        score = torch.sum(x_e * x_t, dim=-1)
        if self.calibrator is not None:
            score = self.calibrator(score)

        return score


def read_data(
    v_file: PathLike,
    key_file: PathLike,
    enroll_file: PathLike,
    seg_part_idx: int,
    num_seg_parts: int,
) -> Tuple[TrialKey, np.ndarray]:
    """Read enrollment vectors and trial key data.

    Args:
        v_file: Feature rspecifier containing enrollment x-vectors.
        key_file: Trial key file with target/non-target labels.
        enroll_file: Enrollment mapping from model id to vector key.
        seg_part_idx: Current split index (1-based) for test partitioning.
        num_seg_parts: Number of test partitions.
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


def init_model(
    model_path: PathLike,
    embed_layer: Optional[int],
    cal_file: Optional[PathLike],
    threshold: float,
    **kwargs: Any,
) -> MyModel:
    """Initialize verification model and optional score calibrator.

    Args:
        model_path: Path to x-vector model checkpoint.
        embed_layer: Optional embedding layer index for extraction.
        cal_file: Optional calibration model path.
        threshold: Verification threshold used for attack decisions.
        **kwargs: Parsed configuration dictionary with feature settings.
    """
    feat_args = AF.filter_args(**kwargs["feats"])
    logging.info("feat args={}".format(feat_args))
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=False, **feat_args)
    logging.info("feat-extractor={}".format(feat_extractor))

    logging.info("loading model {}".format(model_path))
    xvector_model = TML.load(model_path)
    xvector_model.freeze()
    logging.info("xvector-model={}".format(xvector_model))

    calibrator = None
    if cal_file is not None:
        logging.info("loading calibration params {}".format(cal_file))
        lr = LR.load(cal_file)
        # subting the threshold here will put the decision threshold in 0
        # some attacks use thr=0 to decide if the attack is succesful
        calibrator = Calibrator(lr.A[0, 0], lr.b[0] - threshold)

    model = MyModel(feat_extractor, xvector_model, embed_layer, calibrator)
    model.eval()
    return model


def init_attack_factory(
    wav_scale: float = 1.0, **kwargs: Any
) -> RandomAttackFactory:
    """Initialize random attack factory for waveform perturbations.

    Args:
        wav_scale: Waveform scaling used for epsilon and clipping bounds.
        **kwargs: Parsed configuration dictionary with attack settings.
    """
    attacks_args = RandomAttackFactory.filter_args(**kwargs["attacks"])
    extra_args = {
        "eps_scale": wav_scale,
        "range_min": -wav_scale,
        "range_max": wav_scale,
        "loss": nn.functional.binary_cross_entropy_with_logits,
        "time_dim": 1,
    }
    attacks_args.update(extra_args)

    logging.info("attacks args={}".format(attacks_args))
    attack_factory = RandomAttackFactory(**attacks_args)
    return attack_factory


def init_device(use_gpu: bool) -> torch.device:
    """Initialize compute device.

    Args:
        use_gpu: If ``True``, request one GPU device.
    """
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def skip_attack(is_target: bool, p_tar_attack: float, p_non_attack: float) -> bool:
    """Randomly decide whether to skip attack generation for a trial.

    Args:
        is_target: ``True`` for target trials, ``False`` for non-target.
        p_tar_attack: Probability of attacking a target trial.
        p_non_attack: Probability of attacking a non-target trial.
    """
    p = torch.rand(1).item()
    if is_target:
        if p > p_tar_attack:
            return True
    else:
        if p > p_non_attack:
            return True

    return False


def generate_attacks(
    v_file: PathLike,
    key_file: PathLike,
    enroll_file: PathLike,
    test_wav_file: PathLike,
    vad_spec: Optional[PathLike],
    vad_path_prefix: Optional[PathLike],
    model_path: PathLike,
    embed_layer: Optional[int],
    cal_file: Optional[PathLike],
    threshold: float,
    output_wav_dir: PathLike,
    attack_info_file: PathLike,
    attack_tag: str,
    p_tar_attack: float,
    p_non_attack: float,
    save_failed: bool,
    use_gpu: bool,
    seg_part_idx: int,
    num_seg_parts: int,
    random_seed: int,
    **kwargs: Any,
) -> None:
    """Generate adversarial waveforms for verification trials.

    Args:
        v_file: Feature rspecifier with enrollment x-vectors.
        key_file: Trial key file with target/non-target labels.
        enroll_file: Enrollment mapping file.
        test_wav_file: Test audio recordings specifier.
        vad_spec: Optional VAD specifier.
        vad_path_prefix: Optional path prefix for VAD entries.
        model_path: Path to x-vector model checkpoint.
        embed_layer: Optional embedding layer index.
        cal_file: Optional calibration model path.
        threshold: Decision threshold for calibrated scores.
        output_wav_dir: Directory where adversarial waveforms are saved.
        attack_info_file: YAML path for generated attack metadata.
        attack_tag: Tag appended to generated adversarial keys.
        p_tar_attack: Probability of attacking target trials.
        p_non_attack: Probability of attacking non-target trials.
        save_failed: If ``True``, also save failed attacks.
        use_gpu: If ``True``, run generation on GPU.
        seg_part_idx: Current split index (1-based) for test partitioning.
        num_seg_parts: Number of test partitions.
        random_seed: Base random seed used for reproducibility.
        **kwargs: Parsed configuration dictionary.
    """
    device = init_device(use_gpu)
    model = init_model(model_path, embed_layer, cal_file, threshold, **kwargs)
    model.to(device)

    tar = torch.as_tensor([1], dtype=torch.float).to(device)
    non = torch.as_tensor([0], dtype=torch.float).to(device)

    logging.info("loading key and enrollment x-vectors")
    key, x_e = read_data(v_file, key_file, enroll_file, seg_part_idx, num_seg_parts)
    x_e = torch.as_tensor(x_e, dtype=torch.get_default_dtype())

    logging.info("opening audio read stream: %s", test_wav_file)
    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(recordings=test_wav_file, **audio_args)
    wav_scale = audio_reader.wav_scale
    kwargs["wav_scale"] = wav_scale

    logging.info("opening audio write stream: %s", output_wav_dir)
    audio_writer = AW(output_wav_dir, audio_format="flac")

    if vad_spec is not None:
        logging.info("opening VAD stream: %s", vad_spec)
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

    attack_factory = init_attack_factory(**kwargs)
    attacks_info = {}

    for j in range(key.num_tests):
        t1 = time.time()
        logging.info("scoring test utt %s", key.seg_set[j])
        s, fs = audio_reader.read([key.seg_set[j]])
        s = s[0]
        fs = fs[0]
        torch.manual_seed(random_seed + len(s))  # this is to make results reproducible
        s = torch.as_tensor(s[None, :], dtype=torch.get_default_dtype()).to(device)

        if vad_spec is not None:
            vad = v_reader.read([key.seg_set[j]])[0]
            tot_frames = len(vad)
            speech_frames = np.sum(vad)
            vad = torch.as_tensor(vad.astype(bool, copy=False), dtype=torch.bool).to(
                device
            )
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
        for i in range(key.num_models):
            trial_id = "%s-%s" % (key.model_set[i], key.seg_set[j])
            if key.tar[i, j] or key.non[i, j]:
                t3 = time.time()
                if skip_attack(key.tar[i, j], p_tar_attack, p_non_attack):
                    logging.info("skipping attack for tar trial %s", trial_id)
                    continue

                model.x_e = x_e[i : i + 1].to(device)
                with torch.no_grad():
                    score_benign = model(s)

                if key.tar[i, j] and score_benign < 0:
                    logging.info(
                        "target trial %s failed benign classification, skipping...",
                        trial_id,
                    )
                    continue
                elif key.non[i, j] and score_benign > 0:
                    logging.info(
                        "non-target trial %s failed benign classification, skipping...",
                        trial_id,
                    )
                    continue

                attack = attack_factory.sample_attack(model)
                if key.tar[i, j]:
                    t = non if attack.targeted else tar
                else:
                    t = tar if attack.targeted else non

                attack_info = attack.attack_info
                s_adv = attack.generate(s, t).detach()
                with torch.no_grad():
                    # we add the threshold back here to make sure the scores are well calibrated
                    score_adv = model(s_adv)

                t4 = time.time()
                trial_time += t4 - t3
                num_trials += 1
                success = True
                if key.tar[i, j] and score_adv > 0:
                    success = False
                    if not save_failed:
                        logging.info(
                            "attack on target trial %s failed, skipping...", trial_id
                        )
                        continue
                elif key.non[i, j] and score_adv < 0:
                    success = False
                    if not save_failed:
                        logging.info(
                            "attack on non-target trial %s failed benign classification, skipping...",
                            trial_id,
                        )
                        continue
                if success:
                    logging.info("attack on trial %s successful", trial_id)

                stats_ij = compute_stats_adv_attack(s, s_adv)
                stats_ij = [float(stat.detach().cpu().numpy()[0]) for stat in stats_ij]

                s_adv = s_adv.cpu().numpy()[0]
                key_attack = "%s-%s" % (trial_id, attack_tag)
                output_wav = audio_writer.write(key_attack, s_adv, fs)

                attack_info.update(
                    {
                        "attack_tag": attack_tag,
                        "wav_path": output_wav[0],
                        "class_name": "target" if key.tar[i, j] else "non-target",
                        "class_id": int(key.tar[i, j]),
                        "key_benign": trial_id,
                        "enroll": str(key.model_set[i]),
                        "test_benign": str(key.seg_set[j]),
                        "snr": stats_ij[0],
                        "px": stats_ij[1],
                        "pn": stats_ij[2],
                        "x_l2": stats_ij[3],
                        "x_linf": stats_ij[4],
                        "n_l0": stats_ij[5],
                        "n_l2": stats_ij[6],
                        "n_linf": stats_ij[7],
                        "num_samples": s.shape[-1],
                        "success": success,
                    }
                )
                attacks_info[key_attack] = attack_info

        if num_trials > 0:
            trial_time /= num_trials
            t7 = time.time()
            logging.info(
                (
                    "utt %s total-time=%.3f read-time=%.3f trial-time=%.3f n_trials=%d "
                    "rt-factor=%.4f"
                ),
                key.seg_set[j],
                t7 - t1,
                t2 - t1,
                trial_time,
                num_trials,
                num_trials * len(s) / fs / (t7 - t1),
            )

    logging.info("saving attack info to %s", attack_info_file)
    Path(attack_info_file).parent.mkdir(parents=True, exist_ok=True)

    with open(attack_info_file, "w") as f:
        # only save if we have successful attacks
        if attacks_info:
            yaml.dump(attacks_info, f, sort_keys=True)


def main() -> None:
    """Parse CLI arguments and generate adversarial verification attacks."""
    parser = ArgumentParser(
        description="Generate Attacks for speaker verification with x-vectors+cos+calibration"
    )

    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Path to a configuration file.",
    )
    parser.add_argument(
        "--v-file",
        required=True,
        help="Input x-vector rspecifier for enrollment embeddings.",
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="Trial key file with target/non-target labels.",
    )
    parser.add_argument(
        "--enroll-file",
        required=True,
        help="Enrollment file mapping model ids to x-vector keys.",
    )
    parser.add_argument(
        "--test-wav-file",
        required=True,
        help="Audio recordings specifier for test utterances.",
    )
    parser.add_argument(
        "--attack-tag",
        required=True,
        help="Tag appended to generated adversarial utterance keys.",
    )

    AR.add_class_args(parser)
    AF.add_class_args(parser, prefix="feats")

    parser.add_argument(
        "--vad",
        dest="vad_spec",
        default=None,
        help="Optional VAD specifier used to select speech frames.",
    )
    parser.add_argument(
        "--vad-path-prefix",
        dest="vad_path_prefix",
        default=None,
        help="Optional path prefix prepended to VAD SCP paths.",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to x-vector model checkpoint.",
    )
    parser.add_argument(
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "Embedding layer index to extract from; if not set, "
            "use the model default from training."
        ),
    )

    parser.add_argument(
        "--use-gpu",
        default=False,
        action="store_true",
        help="Run attack generation on GPU.",
    )

    parser.add_argument("--cal-file", default=None, help="score calibration file")
    parser.add_argument("--threshold", default=0, type=float, help="decision threshold")

    RandomAttackFactory.add_class_args(parser, prefix="attacks")

    parser.add_argument(
        "--seg-part-idx",
        default=1,
        type=int,
        help="1-based index of the current test partition.",
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
        "--output-wav-dir",
        default=None,
        help="Directory where adversarial waveforms are written.",
    )
    parser.add_argument(
        "--attack-info-file",
        default=None,
        help="YAML file where metadata for generated attacks is saved.",
    )
    parser.add_argument(
        "--random-seed", default=1234, type=int, help="Base random seed."
    )

    parser.add_argument(
        "--p-tar-attack",
        type=float,
        default=1,
        help=("probability of generating an attack for a target trial"),
    )
    parser.add_argument(
        "--p-non-attack",
        type=float,
        default=1,
        help=("probability of generating an attack for a non-target trial"),
    )
    parser.add_argument(
        "--save-failed",
        default=False,
        action="store_true",
        help="Save failed attacks as well.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    generate_attacks(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
