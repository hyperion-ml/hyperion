#!/usr/bin/env python
# prepare_common_voice.py --corpus-dir /export/corpora6/LRE/CommonVoice2020 --output-dir data/cv --map-langs-to-lre-codes --target-fs 8000
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
import logging
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from hyperion.hyp_defs import config_logger

en_map = {"indian": "eng-ine"}
fr_map = {
    "france": "fra-fra",
    "canada": "fra-can",
    "algeria": "fra-ntf",
    "morocco": "fra-ntf",
    "tunisia": "fra-ntf",
}

lre_map = {
    "en": en_map,
    "fr": fr_map,
}


def make_kaldi(df, wav_dir, output_dir, target_fs):
    # make wav.scp
    logging.info("making wav.scp")
    list_dir = output_dir / "lists_cat"
    list_dir.mkdir(parents=True, exist_ok=True)
    for r in range(len(df)):
        file_list = df.iloc[r].file_lists
        with open(list_dir / f"{df.iloc[r].id}.txt", "w") as f:
            for fn in file_list:
                f.write("file %s\n" % fn)

    with open(output_dir / "wav.scp", "w") as f:
        for _, row in df.iterrows():
            segment_id = row["id"]
            filename = list_dir / f"{segment_id}.txt"
            if target_fs != 16000:
                wav = f"ffmpeg -f concat -safe 0 -i {filename} -acodec pcm_s16le -ar {target_fs} -f wav - |"
            else:
                wav = f"ffmpeg -f concat -safe 0 -i {filename} -acodec pcm_s16le -f wav - |"

            f.write("{} {}\n".format(segment_id, wav))

    # Kaldi data directory files
    # utt2xxx files
    logging.info("saving Kaldi utt2xxx files")
    columns = [
        "id",
        "id",
        "language",
    ]
    files = [
        "utt2spk",
        "spk2utt",
        "utt2lang",
    ]
    for c, f in zip(columns, files):
        output_file = output_dir / f
        if c in df:
            df.to_csv(output_file,
                      sep=" ",
                      columns=["id", c],
                      header=False,
                      index=False)


def prepare_common_voice(corpus_dir, output_dir, lang, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir
    my_map = lre_map[lang]
    df = pd.read_csv(corpus_dir / lang / "validated.tsv", sep="\t")
    mask = None
    for dialect in my_map.keys():
        mask_d = df["accent"] == dialect
        if mask is None:
            mask = mask_d
        else:
            mask = np.logical_or(mask, mask_d)

    df = df.loc[mask]
    files = df["path"]
    files = [corpus_dir / lang / "clips" / f for f in df["path"]]
    langs = [my_map[l] for l in df["accent"]]
    ids = ["{}-{}".format(l, Path(f).stem) for f, l in zip(files, langs)]
    df = pd.DataFrame({
        "id": ids,
        "language": langs,
        "filename": files,
        "speaker": df["client_id"]
    })

    # sort by speaker, id
    df.sort_values(by=["speaker", "id"], inplace=True)

    file_lists = []
    file_list = []
    seg_count = 0
    prev_spk = ""
    cat_segs = []
    cur_seg = 0
    for r in range(len(df)):
        row = df.iloc[r]
        if seg_count == 5 or (row.speaker != prev_spk and seg_count > 0):
            file_lists.append(file_list)
            cat_segs.append(cur_seg)
            file_list = []
            seg_count = 0
            cur_seg = r

        file_list.append(row.filename)
        seg_count += 1
        prev_spk = row.speaker

    if file_list:
        file_lists.append(file_list)
        cat_segs.append(cur_seg)

    df_cat = df.iloc[cat_segs].drop(["filename"], axis=1)
    df_cat["file_lists"] = file_lists

    df_cat["sample_coding"] = "pcm"
    df_cat["source"] = "afv"
    df_cat["corpus_id"] = "cv"
    df_cat["sample_rate"] = target_fs

    # sort by segment id
    df_cat.sort_values(by="id", inplace=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "segments.csv"
    logging.info("saving %s", output_file)
    df_cat.drop(["file_lists"], axis=1).to_csv(output_file,
                                               sep=",",
                                               index=False)

    make_kaldi(df_cat, wav_dir, output_dir, target_fs)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Prepares Common Voice Accents for training in LRE22")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument("--lang",
                        default="en",
                        choices=["en", "fr"],
                        help="languages")

    parser.add_argument("--target-fs",
                        default=8000,
                        type=int,
                        help="Target sampling frequency")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()
    prepare_common_voice(**namespace_to_dict(args))