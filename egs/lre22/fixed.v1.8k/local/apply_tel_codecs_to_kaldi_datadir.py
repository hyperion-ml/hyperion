#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
import logging
from pathlib import Path
import glob
import shutil
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import RecordingSet, SegmentSet

valid_codecs = ["gsm", "g711mu", "g711a", "g722", "g723_1", "g726", "opus"]

sox_options = {"gsm": "-r 8000 -e gsm-full-rate -t gsm"}
ffmpeg_options = {
    "g711a": "-ar 8000 -acodec pcm_alaw -f wav",
    "g711mu": "-ar 8000 -acodec pcm_mulaw -f wav",
    "g722": "-ar 8000 -acodec g722 -f wav",
    "g723_1": "-ar 8000 -acodec g723_1 -b:a 6300 -f wav",
    "g726": "-ar 8000 -acodec g726 -f wav",
    "opus": "-ar 8000 -acodec libopus -application voip -f opus",
}


def apply_sox_codec(storage_path, codec):

    option = sox_options[codec]
    storage_path = storage_path.rstrip()
    if storage_path[-1] == "|":
        storage_path = f"{storage_path} sox -t wav - {option} - |"
    else:
        storage_path = f"sox {storage_path} {option} - |"

    storage_path = f"{storage_path} sox {option} - -t wav -e signed-integer -b 16 - |"
    return storage_path


def apply_ffmpeg_codec(storage_path, codec, g726_css, opus_brs, rng):

    option = ffmpeg_options[codec]
    if codec == "g726":
        code_size = rng.choice(g726_css)
        option = f"{option} -code_size {code_size}"
    elif codec == "opus":
        br = rng.choice(opus_brs)
        option = f"{option} -b:a {br}"

    storage_path = storage_path.rstrip()
    if storage_path[-1] == "|":
        storage_path = f"{storage_path} ffmpeg -i - {option} - |"
    else:
        storage_path = f"ffmpeg -i {storage_path} {option} - |"

    storage_path = f"{storage_path} ffmpeg -i - -ar 8000 -c:a pcm_s16le -f wav - |"
    return storage_path


def apply_codec(storage_path, codec, g726_css, opus_brs, rng):

    if codec in ["gsm"]:
        storage_path = apply_sox_codec(storage_path, codec)
    else:
        storage_path = apply_ffmpeg_codec(storage_path, codec, g726_css,
                                          opus_brs, rng)

    return storage_path


def apply_codecs(
    input_dir,
    output_dir,
    codecs,
    keep_orig,
    g726_min_code_size,
    opus_brs,
    seed,
    verbose,
):
    config_logger(verbose)
    logging.info("Applying codecs %s -> %s", input_dir, output_dir)
    rng = np.random.RandomState(seed=seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    g726_css = list(range(g726_min_code_size, 6))
    logging.info("making wav.scp")
    recs = RecordingSet.load(input_dir / "wav.scp")
    recs["orig_id"] = recs["id"]
    if keep_orig:
        recs_orig = recs.clone()

    codec_idx = 0
    ids = []
    s_paths = []
    for i in tqdm(range(len(recs))):
        t1 = time.time()
        row = recs.iloc[i]
        t2 = time.time()
        codec_i = codecs[codec_idx % len(codecs)]
        codec_idx += 1
        t3 = time.time()
        # recs.loc[row.id, "id"] = f"{row.id}-{codec_i}"
        ids.append(f"{row.id}-{codec_i}")
        t4 = time.time()
        sp = apply_codec(row["storage_path"], codec_i, g726_css, opus_brs, rng)

        t5 = time.time()
        # recs.loc[row.id, "storage_path"] = sp
        s_paths.append(sp)
        t6 = time.time()

    recs["id"] = ids
    recs["storage_path"] = s_paths

    mapping = recs[["orig_id", "id"]]
    mapping.set_index("orig_id", inplace=True, drop=False)
    if keep_orig:
        recs = RecordingSet.merge(recs_orig, recs)
        recs.sort()

    logging.info("making utt2orig_utt")
    recs[["id", "orig_id"]].to_csv(output_dir / "utt2orig_utt",
                                   sep=" ",
                                   header=False,
                                   index=False)

    recs.save(output_dir / "wav.scp")
    u2x_files = []
    for pattern in ["utt2*", "vad.scp", "feats.scp"]:
        files_p = glob.glob(str(input_dir / pattern))
        u2x_files.extend(files_p)

    for f in u2x_files:
        logging.info("making %s", Path(f).name)
        u2x = SegmentSet.load(f)
        if keep_orig:
            u2x_orig = u2x.clone()

        u2x["id"] = mapping.loc[u2x["id"], "id"]
        if keep_orig:
            u2x = SegmentSet.merge(u2x_orig, u2x)
            u2x.sort()

        output_file = output_dir / Path(f).name
        u2x.save(output_file)

    spk_files = glob.glob(str(input_dir / "spk2gender"))
    for f in spk_files:
        logging.info("making %s", Path(f).name)
        output_file = output_dir / Path(f).name
        shutil.copy2(f, output_file)

    logging.info("making utt2spk")
    u2s = SegmentSet.load(output_dir / "utt2spk")
    spks = u2s["class_id"].unique()
    df_spk = u2s.df.sort_values(by="class_id")
    df_spk.set_index("class_id", inplace=True)

    with open(output_dir / "spk2utt", "w") as f:
        for spk in spks:
            seg_ids = df_spk.loc[spk, "id"]
            if isinstance(seg_ids, list):
                seg_ids = " ".join(seg_ids)
            f.write(f"{spk} {seg_ids}\n")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Apply telephone codecs to kaldi data dir")
    parser.add_argument("--input-dir",
                        required=True,
                        help="Path to the original kaldi dataset")

    parser.add_argument("--output-dir",
                        required=True,
                        help="Codec augmented directory")
    parser.add_argument(
        "--codecs",
        default=valid_codecs,
        nargs="+",
        choices=valid_codecs,
        help="List of codecs to apply",
    )
    parser.add_argument(
        "--g726-min-code-size",
        default=2,
        choices=[2, 3, 4, 5],
        help="minimum code-size for g726",
    )
    parser.add_argument(
        "--opus-brs",
        default=[4500, 5500, 7700, 9500, 12500, 16000, 32000],
        nargs="+",
        help="opus codec bit rates",
    )
    parser.add_argument("--keep-orig", default=False, action=ActionYesNo)
    parser.add_argument("--seed", default=1234, help="random seed")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    apply_codecs(**namespace_to_dict(args))
