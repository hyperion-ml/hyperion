"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as tat
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

# from hyperion.torch import TorchModelLoader as TML
from hyperion.torch import TorchModel
from hyperion.torch.utils import open_device
from hyperion.utils import Utt2Info

resamplers = {}


def get_resampler(source_fs, target_fs):
    if source_fs in resamplers:
        return resamplers[source_fs]

    resampler = tat.Resample(
        int(source_fs),
        int(target_fs),
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="kaiser_window",
        beta=14.769656459379492,
    )
    resampler_f = lambda x: resampler(torch.from_numpy(x)).numpy()
    resamplers[source_fs] = resampler_f
    return resampler_f


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus=%d", num_gpus)
    device = open_device(num_gpus=num_gpus)
    return device


def load_model(model_path, device):
    logging.info("loading model %s", model_path)
    model = TorchModel.auto_load(model_path)
    logging.info(f"xvector-model={model}")
    model.to(device)
    model.eval()
    return model


def augment(key0, x0, augmenter, aug_df, aug_id):
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

        aug_df.append(pd.DataFrame(aug_df_row, index=[0]))

    return key, x


def select_random_chunk(key, x, fs, min_utt_length, max_utt_length, rng):
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
    recordings_file,
    output_spec,
    vad_spec,
    write_speech_dur,
    vad_path_prefix,
    model_path,
    chunk_length,
    random_utt_length,
    min_utt_length,
    max_utt_length,
    aug_cfg,
    num_augs,
    aug_info_path,
    use_gpu,
    **kwargs,
):
    rng = np.random.default_rng(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    model = load_model(model_path, device)

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

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output stream: %s with args=%s", output_spec, str(ar_args))
    with DWF.create(output_spec, metadata_columns=metadata_columns) as writer:
        logging.info(f"opening input stream: {recordings_file} with args={ar_args}")
        with AR(recordings_file, **ar_args) as reader:
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
                fs = fs[0]
                t2 = time.time()
                if fs != model.sample_frequency:
                    resampler = get_resampler(fs, model.sample_frequency)
                    x0 = resampler(x0)

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
                        if vad_spec is not None:
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


def main():
    parser = ArgumentParser(
        description="""Extracts x-vectors from waveform computing acoustic features on the fly"""
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--recordings-file", required=True)
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
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_xvector_logits(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
