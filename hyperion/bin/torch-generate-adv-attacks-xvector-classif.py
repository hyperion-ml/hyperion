#!/usr/bin/env python
"""
  Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import AudioWriter as AW
from hyperion.utils import Utt2Info, TrialNdx
from hyperion.io import VADReaderFactory as VRF

from hyperion.torch.utils import open_device
from hyperion.torch import TorchModelLoader as TML
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils.misc import l2_norm, compute_stats_adv_attack

from hyperion.torch.adv_attacks import RandomAttackFactory


def read_utt_list(list_file, class2int_file, part_idx, num_parts):
    logging.info("reading utt list %s" % (list_file))
    utt_list = Utt2Info.load(list_file)
    utt_list = utt_list.split(part_idx, num_parts)
    logging.info("reading class2int-file %s" % (class2int_file))
    class_info = pd.read_csv(class2int_file, header=None, sep=" ")
    class2idx = {str(k): i for i, k in enumerate(class_info[0])}
    class_idx = np.array([class2idx[k] for k in utt_list.info], dtype=int)
    keys = utt_list.key
    class_names = utt_list.info
    return keys, class_names, class_idx


class MyModel(nn.Module):
    def __init__(self, feat_extractor, xvector_model):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.xvector_model = xvector_model
        self.vad = None

    def forward(self, s):
        f = self.feat_extractor(s)
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


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_model(model_path, **kwargs):
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
    xvector_model = TML.load(model_path)
    xvector_model.freeze()
    logging.info("xvector-model={}".format(xvector_model))

    model = MyModel(feat_extractor, xvector_model)
    model.eval()
    return model


def init_attack_factory(wav_scale=1, **kwargs):
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


def select_random_chunk(key, s, fs, min_utt_length, max_utt_length):
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
    wav_file,
    list_file,
    vad_spec,
    vad_path_prefix,
    class2int_file,
    model_path,
    output_wav_dir,
    attack_info_file,
    attack_tag,
    random_utt_length,
    min_utt_length,
    max_utt_length,
    random_seed,
    p_attack,
    save_failed,
    save_benign,
    use_gpu,
    part_idx,
    num_parts,
    **kwargs
):

    device = init_device(use_gpu)
    model = init_model(model_path, **kwargs)
    model.to(device)

    logging.info("opening audio read stream: %s" % (wav_file))
    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(wav_file)
    wav_scale = audio_reader.wav_scale

    logging.info("opening audio write stream: %s" % (output_wav_dir))
    audio_writer = AW(output_wav_dir, audio_format="flac")

    if vad_spec is not None:
        logging.info("opening VAD stream: %s" % (vad_spec))
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix, scp_sep=" ")

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

        torch.manual_seed(
            random_seed + int(s[0])
        )  # this is to make results reproducible
        p = torch.rand(1).item()
        if p > p_attack:
            logging.info("skipping attack for utt %s" % (key))
            continue

        if random_utt_length:
            s = select_random_chunk(key, s, fs, min_utt_length, max_utt_length)

        if save_benign:
            s_benign = s

        s = torch.as_tensor(s[None, :], dtype=torch.get_default_dtype()).to(device)
        target = torch.as_tensor([class_id], dtype=torch.long).to(device)
        if vad_spec is not None:
            vad = v_reader.read([key.seg_set[j]])[0]
            tot_frames = len(vad)
            speech_frames = np.sum(vad)
            vad = torch.as_tensor(vad.astype(np.bool, copy=False), dtype=torch.bool).to(
                device
            )
            model.vad = vad
            logging.info(
                "utt %s detected %d/%d (%.2f %%) speech frames"
                % (
                    key.seg_set[j],
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
            logging.info("utt %s failed benign classification, skipping..." % (key))
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


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Generate Attacks for speaker classification with x-vectors"
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--wav-file", required=True)
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--class2int-file", required=True)
    parser.add_argument("--attack-tag", required=True)

    AR.add_class_args(parser)
    AF.add_class_args(parser, prefix="feats")

    parser.add_argument("--vad", dest="vad_spec", default=None)
    parser.add_argument(
        "--vad-path-prefix",
        dest="vad_path_prefix",
        default=None,
        help=("scp file_path prefix for vad"),
    )

    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )

    RandomAttackFactory.add_class_args(parser, prefix="attacks")

    parser.add_argument("--part-idx", default=1, type=int, help=("part index"))
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
        "--output-wav-dir", default=None, help="output path of adv signals"
    )
    parser.add_argument(
        "--attack-info-file",
        default=None,
        help="output path of to save information about the generated attacks",
    )
    parser.add_argument(
        "--random-seed", default=1234, type=int, help="random seed for pytorch"
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
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    generate_attacks(**namespace_to_dict(args))
