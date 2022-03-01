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

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.io import RandomAccessAudioReader as AR
from hyperion.io import AudioWriter as AW
from hyperion.utils import Utt2Info, TrialNdx, TrialKey, TrialScores
from hyperion.utils.list_utils import ismember
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.classifiers import BinaryLogisticRegression as LR

from hyperion.torch.utils import open_device
from hyperion.torch.layers import LinBinCalibrator as Calibrator
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch.utils.misc import l2_norm, compute_stats_adv_attack
from hyperion.torch import TorchModelLoader as TML

from art.classifiers import PyTorchClassifier
from hyperion.torch.adv_attacks.art_attack_factory import (
    ARTAttackFactory as AttackFactory,
)


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_feats(**kwargs):
    feat_args = AF.filter_args(**kwargs["feats"])
    logging.info("feat args={}".format(feat_args))
    logging.info("initializing feature extractor")
    feat_extractor = AF(trans=False, **feat_args)
    logging.info("feat-extractor={}".format(feat_extractor))
    feat_extractor.eval()
    return feat_extractor


def load_model(model_path):
    logging.info("loading model {}".format(model_path))
    model = TML.load(model_path)
    logging.info("xvector-model={}".format(model))
    model.eval()
    return model


def load_calibrator(cal_file):
    logging.info("loading calibration params {}".format(cal_file))
    lr = LR.load(cal_file)
    calibrator = Calibrator(lr.A[0, 0], lr.b[0])
    calibrator.eval()
    return calibrator


def read_data(v_file, key_file, enroll_file, seg_part_idx, num_seg_parts):

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
    def __init__(
        self,
        feat_extractor,
        xvector_model,
        embed_layer=None,
        calibrator=None,
        threshold=0,
    ):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.xvector_model = xvector_model
        self.x_e = None
        self.vad_t = None
        self.embed_layer = embed_layer
        self.calibrator = calibrator
        self.threshold = threshold

    def forward(self, s_t):
        f_t = s_t
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
        tar_score = torch.sum(x_e * x_t, dim=-1, keepdim=True)
        if self.calibrator is not None:
            score = self.calibrator(tar_score)

        non_score = self.threshold + 0 * tar_score
        score = torch.cat((non_score, tar_score), dim=-1)  # .unsqueeze(0)
        return score


def eval_cosine_scoring(
    v_file,
    key_file,
    enroll_file,
    test_wav_file,
    vad_spec,
    vad_path_prefix,
    model_path,
    embed_layer,
    score_file,
    stats_file,
    cal_file,
    threshold,
    save_adv_wav,
    save_adv_wav_path,
    max_test_length,
    use_gpu,
    seg_part_idx,
    num_seg_parts,
    **kwargs
):

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

    tar = np.asarray([1], dtype=np.int)
    non = np.asarray([0], dtype=np.int)

    logging.info("loading key and enrollment x-vectors")
    key, x_e = read_data(v_file, key_file, enroll_file, seg_part_idx, num_seg_parts)
    x_e = torch.as_tensor(x_e, dtype=torch.get_default_dtype())

    audio_args = AR.filter_args(**kwargs)
    audio_reader = AR(test_wav_file)
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
        v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix, scp_sep=" ")

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
        logging.info("scoring test utt %s" % (key.seg_set[j]))
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
            vad = torch.as_tensor(vad.astype(np.bool, copy=False), dtype=torch.bool).to(
                device
            )
            model.vad_t = vad
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

        trial_time = 0
        num_trials = 0
        model_art = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=[1, s.shape[1]],
            nb_classes=2,
            clip_values=(-wav_scale, wav_scale),
            device_type=device_type,
        )

        attack_args["num_samples"] = s.shape[-1]
        attack = AttackFactory.create(model_art, **attack_args)
        for i in range(key.num_models):
            if key.tar[i, j] or key.non[i, j]:
                t3 = time.time()
                model.x_e = x_e[i].to(device)
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
    logging.info("saving scores to %s" % (score_file))
    s = TrialScores(
        key.model_set, key.seg_set, scores, score_mask=np.logical_or(key.tar, key.non)
    )
    s.save_txt(score_file)

    logging.info("saving stats to %s" % (stats_file))
    attack_stats.to_csv(stats_file)


if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            "Eval cosine-scoring given enroll x-vector "
            "and adversarial test wave from ART"
        )
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--v-file", dest="v_file", required=True)
    parser.add_argument("--key-file", dest="key_file", default=None)
    parser.add_argument("--enroll-file", dest="enroll_file", required=True)
    parser.add_argument("--test-wav-file", required=True)

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
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "classifier layer to get the embedding from,"
            "if None the layer set in training phase is used"
        ),
    )

    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )

    AttackFactory.add_class_args(parser, prefix="attack")

    parser.add_argument("--seg-part-idx", default=1, type=int, help=("test part index"))
    parser.add_argument(
        "--num-seg-parts",
        default=1,
        type=int,
        help=(
            "number of parts in which we divide the test list "
            "to run evaluation in parallel"
        ),
    )

    parser.add_argument("--score-file", dest="score_file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    parser.add_argument(
        "--save-adv-wav",
        default=False,
        action="store_true",
        help="save adversarial signals to disk",
    )
    parser.add_argument(
        "--save-adv-wav-path", default=None, help="output path of adv signals"
    )

    parser.add_argument(
        "--stats-file", default=None, help="output path of to save stats of adv signals"
    )

    parser.add_argument("--cal-file", default=None, help="score calibration file")
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
