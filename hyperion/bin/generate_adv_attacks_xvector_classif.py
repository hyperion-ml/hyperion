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
from hyperion.io import VADReaderFactory as VRF
from hyperion.torch import HyperTorchModel
from hyperion.torch.adv_attacks import RandomAttackFactory
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils import open_device
from hyperion.torch.utils.misc import compute_stats_adv_attack, l2_norm
from hyperion.utils import TrialNdx, Utt2Info
from hyperion.utils.misc import PathLike


def read_utt_list(
    list_file: PathLike,
    class2int_file: PathLike,
    part_idx: int,
    num_parts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read utterance list and map class names to integer ids.

    Args:
        list_file: Path to utterance list with class names.
        class2int_file: Path to class-to-index mapping file.
        part_idx: Current split index (1-based).
        num_parts: Number of splits used for parallel processing.
    """
    logging.info("reading utt list %s", list_file)
    utt_list = Utt2Info.load(list_file)
    utt_list = utt_list.split(part_idx, num_parts)
    logging.info("reading class2int-file %s", class2int_file)
    class_info = pd.read_csv(class2int_file, header=None, sep=" ")
    class2idx = {str(k): i for i, k in enumerate(class_info[0])}
    class_idx = np.array([class2idx[k] for k in utt_list.info], dtype=int)
    keys = utt_list.key
    class_names = utt_list.info
    return keys, class_names, class_idx


class MyModel(nn.Module):
    """Wrapper model combining feature extraction and x-vector classification."""

    def __init__(self, feat_extractor: AF, xvector_model: nn.Module) -> None:
        """Initialize attack-time classifier wrapper.

        Args:
            feat_extractor: Acoustic feature extractor.
            xvector_model: X-vector classification model.
        """
        super().__init__()
        self.feat_extractor = feat_extractor
        self.xvector_model = xvector_model
        self.vad = None

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute classification scores from waveform input.

        Args:
            s: Input waveform tensor.
        """
        f, _ = self.feat_extractor(s)
        if self.vad is not None:
            n_vad_frames = len(self.vad)
            n_feat_frames = f.shape[1]
            if n_vad_frames > n_feat_frames:
                self.vad = self.vad[:n_feat_frames]
            elif n_vad_frames < n_feat_frames:
                f = f[:, :n_vad_frames]

            f = f[:, self.vad]

        f = f.transpose(1, 2).contiguous()
        score = self.xvector_model(f)
        return score


def init_device(use_gpu: bool) -> torch.device:
    """Initialize runtime device for attack generation.

    Args:
        use_gpu: If ``True``, request one GPU device.
    """
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_model(model_path: PathLike, **kwargs: Any) -> MyModel:
    """Initialize attack model wrapper from configuration and checkpoint.

    Args:
        model_path: Path to x-vector model checkpoint.
        **kwargs: Parsed args containing feature-extractor configuration.
    """
    feat_args = AF.filter_args(**kwargs["feats"])
    logging.info("feat args={}".format(feat_args))
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=False, **feat_args)
    logging.info("feat-extractor={}".format(feat_extractor))

    # feat_args = AFF.filter_args(prefix='feats', **kwargs)
    # logging.info('initializing feature extractor args={}'.format(feat_args))
    # feat_extractor = AFF.create(**feat_args)

    # mvn_args = MVN.filter_args(prefix='mvn', **kwargs)
    # mvn = None
    # if mvn_args['norm_mean'] or mvn_args['norm_var']:
    #     logging.info('initializing short-time mvn args={}'.format(mvn_args))
    #     mvn = MVN(**mvn_args)

    logging.info("loading model {}".format(model_path))
    xvector_model = HyperTorchModel.auto_load(model_path)
    xvector_model.freeze()
    logging.info("xvector-model={}".format(xvector_model))

    model = MyModel(feat_extractor, xvector_model)
    model.eval()
    return model


def init_attack_factory(wav_scale: float = 1.0, **kwargs: Any) -> RandomAttackFactory:
    """Initialize random attack factory.

    Args:
        wav_scale: Waveform scale used for attack epsilon and clipping ranges.
        **kwargs: Parsed args containing attack configuration.
    """
    attacks_args = RandomAttackFactory.filter_args(**kwargs["attacks"])
    extra_args = {
        "eps_scale": wav_scale,
        "range_min": -wav_scale,
        "range_max": wav_scale,
        "loss": nn.functional.cross_entropy,
        "time_dim": 1,
    }
    attacks_args.update(extra_args)

    logging.info("attacks args={}".format(attacks_args))
    attack_factory = RandomAttackFactory(**attacks_args)
    return attack_factory


def select_random_chunk(
    key: str,
    s: np.ndarray,
    fs: int,
    min_utt_length: int,
    max_utt_length: int,
) -> np.ndarray:
    """Randomly crop waveform between min and max duration in seconds.

    Args:
        key: Utterance key used for logging.
        s: Waveform samples.
        fs: Sampling rate in Hz.
        min_utt_length: Minimum random duration in seconds.
        max_utt_length: Maximum random duration in seconds.
    """
    utt_length = torch.randint(
        low=min_utt_length * fs, high=max_utt_length * fs + 1, size=(1,)
    ).item()
    if utt_length < len(s):
        first_sample = torch.randint(low=0, high=len(s) - utt_length, size=(1,)).item()
        s = s[first_sample : first_sample + utt_length]
        logging.info(
            "extract-random-utt %s of length=%d first-sample=%d"
            % (key, len(s), first_sample)
        )
    return s


def generate_attacks(
    wav_file: PathLike,
    list_file: PathLike,
    vad_spec: Optional[PathLike],
    vad_path_prefix: Optional[PathLike],
    class2int_file: PathLike,
    model_path: PathLike,
    output_wav_dir: PathLike,
    attack_info_file: PathLike,
    attack_tag: str,
    random_utt_length: bool,
    min_utt_length: int,
    max_utt_length: int,
    random_seed: int,
    p_attack: float,
    save_failed: bool,
    save_benign: bool,
    use_gpu: bool,
    part_idx: int,
    num_parts: int,
    **kwargs: Any,
) -> None:
    """Generate adversarial waveform attacks for x-vector classification.

    Args:
        wav_file: Input recordings specifier.
        list_file: Input utterance list with class names.
        vad_spec: Optional VAD specifier for frame selection.
        vad_path_prefix: Optional path prefix applied to VAD entries.
        class2int_file: Class-to-index mapping file.
        model_path: X-vector model checkpoint path.
        output_wav_dir: Output directory for attacked waveforms.
        attack_info_file: Output YAML file with attack metadata/statistics.
        attack_tag: Tag appended to attacked utterance keys.
        random_utt_length: Whether to attack a random crop of each utterance.
        min_utt_length: Minimum random crop duration in seconds.
        max_utt_length: Maximum random crop duration in seconds.
        random_seed: Base seed for random operations.
        p_attack: Probability of generating an attack for an utterance.
        save_failed: Whether to save failed attacks.
        save_benign: Whether to save benign copies alongside attacks.
        use_gpu: Whether to run attack generation on GPU.
        part_idx: Current split index (1-based).
        num_parts: Number of splits for parallel processing.
        **kwargs: Additional parsed args for reader/features/attacks.
    """
    device = init_device(use_gpu)
    model = init_model(model_path, **kwargs)
    model.to(device)

    logging.info("opening audio read stream: %s", wav_file)
    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(recordings=wav_file, **audio_args)
    wav_scale = audio_reader.wav_scale

    logging.info("opening audio write stream: %s", output_wav_dir)
    audio_writer = AW(output_wav_dir, audio_format="flac")

    if vad_spec is not None:
        logging.info("opening VAD stream: %s", vad_spec)
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

    keys, class_names, class_ids = read_utt_list(
        list_file, class2int_file, part_idx, num_parts
    )

    attack_factory = init_attack_factory(**kwargs)
    attacks_info = {}

    for i in range(len(keys)):
        key = keys[i]
        class_id = class_ids[i]

        t1 = time.time()
        logging.info("reading utt %s" % (key))
        s, fs = audio_reader.read([key])
        s = s[0]
        fs = fs[0]

        torch.manual_seed(random_seed + len(s))  # this is to make results reproducible
        p = torch.rand(1).item()
        if p > p_attack:
            logging.info("skipping attack for utt %s", key)
            continue

        if random_utt_length:
            s = select_random_chunk(key, s, fs, min_utt_length, max_utt_length)

        if save_benign:
            s_benign = s

        s = torch.as_tensor(s[None, :], dtype=torch.get_default_dtype()).to(device)
        target = torch.as_tensor([class_id], dtype=torch.long).to(device)
        if vad_spec is not None:
            vad = v_reader.read([key])[0]
            tot_frames = len(vad)
            speech_frames = np.sum(vad)
            vad = torch.as_tensor(vad.astype(bool, copy=False), dtype=torch.bool).to(
                device
            )
            model.vad = vad
            logging.info(
                "utt %s detected %d/%d (%.2f %%) speech frames"
                % (
                    key,
                    speech_frames,
                    tot_frames,
                    speech_frames / tot_frames * 100,
                )
            )

        t2 = time.time()
        with torch.no_grad():
            score_benign = model(s)

        _, pred = torch.max(score_benign, dim=1)
        if pred[0] != class_id:
            logging.info("utt %s failed benign classification, skipping...", key)
            continue

        t3 = time.time()
        attack = attack_factory.sample_attack(model)
        attack_info = attack.attack_info
        s_adv = attack.generate(s, target).detach()
        t4 = time.time()
        with torch.no_grad():
            score_adv = model(s_adv)
        t5 = time.time()

        _, pred = torch.max(score_adv, dim=1)
        success = False
        if pred[0] != class_id:
            success = True

        if success or save_failed:
            key_attack = "%s-%s" % (key, attack_tag)
            logging.info("utt %s attack successful" % (key))

            stats_ij = compute_stats_adv_attack(s, s_adv)
            stats_ij = [float(stat.detach().cpu().numpy()[0]) for stat in stats_ij]

            s_adv = s_adv.cpu().numpy()[0]
            wav_attack = audio_writer.write(key_attack, s_adv, fs)[0]
            if save_benign:
                key_benign = "%s-benign" % (key_attack)
                wav_benign = audio_writer.write(key_benign, s_benign, fs)[0]
            else:
                key_benign = key
                wav_benign = ""

            attack_info.update(
                {
                    "attack_tag": attack_tag,
                    "wav_path": wav_attack,
                    "class_name": class_names[i],
                    "class_id": int(class_id),
                    "key_benign": key_benign,
                    "wav_benign": wav_benign,
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

        else:
            logging.info("utt %s attack failed, skipping..." % (key))

        t6 = time.time()
        logging.info(
            (
                "utt %s total-time=%.3f read-time=%.3f "
                "eval-benign-time=%.3f attack-time=%.3f eval-attack-time=%3f "
                "rt-factor=%.4f"
            )
            % (
                key,
                t6 - t1,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
                s.shape[1] / fs / (t6 - t1),
            )
        )

    logging.info("saving attack info to %s" % (attack_info_file))
    Path(attack_info_file).parent.mkdir(parents=True, exist_ok=True)

    with open(attack_info_file, "w") as f:
        # only save if we have successful attacks
        if attacks_info:
            yaml.dump(attacks_info, f, sort_keys=True)


def main() -> None:
    """Parse CLI arguments and run attack generation.

    Args:
        None.
    """
    parser = ArgumentParser(
        description="Generate Attacks for speaker classification with x-vectors"
    )

    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    parser.add_argument(
        "--wav-file", required=True, help="input waveform recordings specifier"
    )
    parser.add_argument(
        "--list-file", required=True, help="utterance list with class names"
    )
    parser.add_argument(
        "--class2int-file", required=True, help="class-to-index mapping file"
    )
    parser.add_argument(
        "--attack-tag", required=True, help="tag appended to attacked utterance keys"
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
        "--model-path", required=True, help="x-vector model checkpoint path"
    )
    parser.add_argument(
        "--use-gpu",
        default=False,
        action="store_true",
        help="run attack generation on GPU",
    )

    RandomAttackFactory.add_class_args(parser, prefix="attacks")

    parser.add_argument(
        "--part-idx",
        default=1,
        type=int,
        help="split index (1-based) when processing list in parts",
    )
    parser.add_argument(
        "--num-parts",
        default=1,
        type=int,
        help=(
            "number of parts in which we divide the list "
            "to run evaluation in parallel"
        ),
    )

    parser.add_argument(
        "--output-wav-dir",
        default=None,
        help="output directory for attacked waveforms",
    )
    parser.add_argument(
        "--attack-info-file",
        default=None,
        help="output YAML file with metadata/statistics of generated attacks",
    )
    parser.add_argument(
        "--random-seed", default=1234, type=int, help="random seed for PyTorch"
    )

    parser.add_argument(
        "--random-utt-length",
        default=False,
        action="store_true",
        help="generate attack from a random utterance crop",
    )
    parser.add_argument(
        "--min-utt-length",
        type=int,
        default=5,
        help=("minimum utterance length (in secs) when using random utt length"),
    )
    parser.add_argument(
        "--max-utt-length",
        type=int,
        default=120,
        help=("maximum utterance length (in secs) when using random utt length"),
    )

    parser.add_argument(
        "--p-attack",
        type=float,
        default=1,
        help=("probability of generating an attack for a given utterance"),
    )
    parser.add_argument(
        "--save-failed",
        default=False,
        action="store_true",
        help=("save failed attacks also"),
    )
    parser.add_argument(
        "--save-benign",
        default=False,
        action="store_true",
        help=("save a copy of the benign sample"),
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

    generate_attacks(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
