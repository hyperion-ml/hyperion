#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as tat
import pickle
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
from hyperion.np import HyperNPModel

# from hyperion.torch import TorchModelLoader as TML
from hyperion.torch import HyperTorchModel
from hyperion.torch.utils import open_device
from hyperion.utils import Utt2Info, SegmentSet
from hyperion.np.diarization import DiarAHCPLDA
from hyperion.np.classifiers import BinaryLogisticRegression as LR
from hyperion.utils.vad_utils import vad_timestamps_to_bin_samples


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus=%d", num_gpus)
    device = open_device(num_gpus=num_gpus)
    return device


def load_model(model_path, device):
    logging.info("loading model %s", model_path)
    model = HyperTorchModel.auto_load(model_path)
    logging.info(f"xvector-model={model}")
    model.to(device)
    model.eval()
    return model


def apply_preprocessor(x, source_type, language, preprocessors):

    cond = f"{source_type}_{language}"
    preprocessor_c = preprocessors[cond]
    x_out = preprocessor_c(x)

    return x_out


def extract_xvectors(
    recordings_file,
    segments_file,
    output_spec,
    vad_spec,
    write_speech_dur,
    vad_path_prefix,
    model_path,
    chunk_length,
    embed_layer,
    random_utt_length,
    min_utt_length,
    max_utt_length,
    aug_cfg,
    num_augs,
    aug_info_path,
    win_length,
    win_shift,
    min_cluster_duration,
    preproc_file,
    plda_file,
    calibration_file,
    ahc,
    debug_dir,
    use_gpu,
    **kwargs,
):
    rng = np.random.default_rng(seed=1123581321 + kwargs["part_idx"])
    torch.backends.cudnn.benchmark = False
    device = init_device(use_gpu)
    model = load_model(model_path, device)
    resampler = ResamplerToTargetFreq(model.sample_frequency)
    segments = SegmentSet.load(segments_file)
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
    if preproc_file is not None:
        logging.info("Loading Preprocessor")

        with open(preproc_file, "rb") as f:
            preprocessors = pickle.load(f)

    else:
        preprocessors = None

    if plda_file is not None:
        logging.info("Loading PLDA model")
        plda_model = HyperNPModel.auto_load(plda_file)
    else:
        plda_model = None

    if calibration_file is not None:
        lr = LR.load(calibration_file)
        scale = lr.A[0]
        bias = lr.b + lr.A[1] + lr.A[3]  # source and lang match
        calibrator = lambda x: scale * x + bias
    else:
        calibrator = None

    diarizer = DiarAHCPLDA(
        preproc=None,
        plda_model=plda_model,
        calibrator=calibrator,
        **ahc,
    )
    if debug_dir is None:
        hist_file = None
    else:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(exist_ok=True, parents=True)

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output stream: %s with args=%s", output_spec, str(ar_args))
    with DWF.create(output_spec, metadata_columns=metadata_columns) as writer:
        logging.info(f"opening input stream: {recordings_file} with args={ar_args}")
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
                key = key[0]
                fs = fs[0]
                t2 = time.time()
                if fs != model.sample_frequency:
                    x0, fs = resampler(x0, fs)

                logging.info("processing utt %s", key)
                metadata = {}
                t3 = time.time()
                if segments.loc[key, "source_type"] == "cts":
                    x = x0
                    t4 = time.time()
                    with torch.no_grad():
                        x = torch.tensor(
                            x[None, :], dtype=torch.get_default_dtype()
                        ).to(device)
                        t5 = time.time()
                        tot_samples = x.shape[1]
                        if vad_spec is not None:
                            vad = v_reader.read(key)[0]
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

                        metadata["speech_duration"] = (
                            x.shape[1] / model.sample_frequency
                        )

                        t6 = time.time()
                        if x.shape[1] == 0:
                            y = np.zeros((model.embed_dim,), dtype=float_cpu())
                        else:
                            y = (
                                model.extract_embed(
                                    x,
                                    chunk_length=chunk_length,
                                    embed_layer=embed_layer,
                                )
                                .cpu()
                                .numpy()[0]
                            )
                        y = y[None, :]

                    t7 = time.time()
                    writer.write([key], [y], metadata=metadata)
                    t8 = time.time()
                    read_time = t2 - t1
                    tot_time = read_time + t8 - t3
                    logging.info(
                        (
                            "utt %s total-time=%.3f read-time=%.3f "
                            "vad-time=%.3f embed-time=%.3f write-time=%.3f "
                            "rt-factor=%.2f"
                        ),
                        key,
                        tot_time,
                        read_time,
                        t6 - t5,
                        t7 - t6,
                        t8 - t7,
                        x.shape[1] / fs / tot_time,
                    )
                else:
                    if augmenter:
                        x, aug_info = augmenter(x0, model.sample_frequency)
                        logging.info(
                            "x-aug-length=%d x-length=%d augmentation=%s",
                            len(x),
                            len(x0),
                            str(aug_info),
                        )
                    else:
                        x = x0

                    t4 = time.time()
                    with torch.no_grad():
                        x = torch.tensor(
                            x[None, :], dtype=torch.get_default_dtype()
                        ).to(device)

                        if vad_spec is not None:
                            t_start, t_end = v_reader.read_time_marks([key])
                            print(t_start[0], t_end[0], x.size(1) / 16000)
                        else:
                            t_start = t_end = None

                        t5 = time.time()
                        y, y_lengths, t_start, t_end = model.extract_embed_slidwin(
                            x,
                            win_length=win_length,
                            win_shift=win_shift,
                            vad_t_start=t_start,
                            vad_t_end=t_end,
                            chunk_length=chunk_length,
                        )
                        t6 = time.time()
                        y = y.cpu().numpy()[0]
                        t_start = t_start[0].cpu().numpy()
                        t_end = t_end[0].cpu().numpy()
                        print(
                            "x-vector",
                            y.shape,
                            y_lengths,
                            t_start,
                            t_end,
                            t_start.shape,
                            t_end.shape,
                        )
                        if y.shape[0] > 1:
                            if debug_dir is not None:
                                hist_file = debug_dir / f"{key}.png"
                                print(hist_file, flush=True)

                            if preprocessors is not None:
                                y = apply_preprocessor(
                                    y,
                                    "cts",
                                    segments.loc[key, "language"],
                                    preprocessors,
                                )

                            cluster_ids, t_start, t_end = diarizer(
                                y,
                                t_start,
                                t_end,
                                hist_file=hist_file,
                            )
                        else:
                            cluster_ids = np.zeros((len(t_start),), dtype=int)

                        t7 = time.time()
                        uniq_cluster_ids = np.unique(cluster_ids)
                        print(
                            "diar",
                            cluster_ids,
                            t_start,
                            t_end,
                            len(cluster_ids),
                            len(uniq_cluster_ids),
                        )
                        y = []
                        speech_duration = np.zeros((len(uniq_cluster_ids),))
                        num_accepted_clusters = 0
                        for cluster_i in uniq_cluster_ids:
                            idx = cluster_ids == cluster_i
                            t_start_i = t_start[idx]
                            t_end_i = t_end[idx]
                            vad_i = vad_timestamps_to_bin_samples(
                                t_start_i,
                                t_end_i,
                                model.sample_frequency,
                                max_samples=x.size(-1),
                            )
                            vad_i = torch.as_tensor(vad_i, device=x.device)
                            x_i = x[:, vad_i]
                            duration_i = x_i.size(1) / fs

                            if duration_i < min_cluster_duration:
                                if cluster_i == uniq_cluster_ids[-1]:
                                    vad_i = vad_timestamps_to_bin_samples(
                                        t_start,
                                        t_end,
                                        model.sample_frequency,
                                        max_samples=x.size(-1),
                                    )
                                    vad_i = torch.as_tensor(vad_i, device=x.device)
                                    x_i = x[:, vad_i]
                                    duration_i = x_i.size(1) / fs
                                else:
                                    continue

                            y_i = (
                                model.extract_embed(
                                    x_i,
                                    chunk_length=chunk_length,
                                    embed_layer=embed_layer,
                                )
                                .cpu()
                                .numpy()[0]
                            )
                            y.append(y_i)
                            speech_duration[num_accepted_clusters] = duration_i
                            num_accepted_clusters += 1

                        speech_duration = speech_duration[:num_accepted_clusters]
                        t8 = time.time()
                        y = np.vstack(y)
                        metadata["speech_duration"] = np.mean(speech_duration)
                        writer.write([key], [y], metadata=metadata)

                        t9 = time.time()
                        read_time = t2 - t1
                        tot_time = read_time + t9 - t3
                        logging.info(
                            (
                                "utt %s total-time=%.3f read-time=%.3f "
                                "vad-time=%.3f slidwin-embed=%.3f diar-time=%.3f "
                                "embed-time=%.3f write-time=%.3f "
                                "rt-factor=%.2f"
                            ),
                            key,
                            tot_time,
                            read_time,
                            t5 - t4,
                            t6 - t5,
                            t7 - t6,
                            t8 - t7,
                            t9 - t8,
                            x.shape[1] / fs / tot_time,
                        )

    if write_speech_dur is not None:
        logging.info("writing speech duration in secs to %s", write_speech_dur)
        u2sd = Utt2Info.create(keys, info)
        u2sd.save(write_speech_dur)

    if aug_info_path is not None:
        aug_df = pd.concat(aug_df, ignore_index=True)
        aug_df.to_csv(aug_info_path, index=False, na_rep="n/a")


def main():
    parser = ArgumentParser(
        description="""Extracts x-vectors from waveform computing acoustic features on the fly"""
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--recordings-file", required=True)
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--vad", dest="vad_spec", default=None)
    parser.add_argument("--write-speech-dur", default=None)
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
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "classifier layer to get the embedding from, "
            "if None, it uses layer set in training phase"
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

    parser.add_argument("--output-spec", required=True)
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )
    parser.add_argument("--win-length", default=1.5, type=float)
    parser.add_argument("--win-shift", default=0.25, type=float)
    DiarAHCPLDA.add_class_args(parser, prefix="ahc")
    parser.add_argument("--preproc-file", default=None)
    parser.add_argument("--plda-file", default=None)
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--min-cluster-duration", default=2.0, type=float)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_xvectors(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
