#!/usr/bin/env python
# prepare_data.py --corpus-dir /export/corpora6/LRE/FLEURS2022 --output-dir data/fleurs --map-langs-to-lre-codes --target-fs 8000
# prepare_data.py --corpus-dir /export/corpora6/LRE/Lwazi2009 --output-dir data/lwazi --map-langs-to-lre-codes --target-fs 8000
# prepare_data.py --corpus-dir /export/corpora6/LRE/NCHLT2014 --output-dir data/nchlt --map-langs-to-lre-codes --target-fs 8000
# prepare_data.py --corpus-dir /export/corpora6/LRE/AMMI2020 --output-dir data/ammi --map-langs-to-lre-codes --target-fs 8000
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

lre_map = {
    "afr": "afr-afr",
    "ndb": "nbl-nbl",
    "oro": "orm-orm",
    "tso": "tso-tso",
    "ven": "ven-ven",
    "xho": "xho-xho",
    "zul": "zul-zul",
    "tig": "tir-tir",
    "sae": "eng-ens",
    "ine": "eng-iaf",
    "tun": "ara-aeb",
    "alg": "ara-arq",
    "lib": "ara-ayl",
    "naf": "fra-ntf"
}

buggy_files = {
    "200630-192328_tir_c85_elicit_7",
    "200701-120846_tir_c85_elicit_35",
    "200701-133352_tir_c85_elicit_57",
    "200701-134903_tir_c85_elicit_19",
    "200701-134903_tir_c85_elicit_32",
    "200701-234652_tir_c85_elicit_78",
    "200702-083859_tir_c85_elicit_18",
    "200702-125252_tir_c85_elicit_46",
    "200702-161120_tir_c85_elicit_4",
    "200702-161120_tir_c85_elicit_7",
    "200702-172026_tir_c85_elicit_31",
    "200702-182933_tir_c85_elicit_133",
    "200702-182933_tir_c85_elicit_88",
    "200702-193310_tir_c85_elicit_2",
    "200702-194850_tir_c85_elicit_88",
    "200702-200911_tir_c85_elicit_171",
}


def map_to_lre(langs):
    return [
        lre_map[l] if l in lre_map else "{}-{}".format(l, l) for l in langs
    ]


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


def prepare_data(corpus_dir, output_dir, remove_langs, map_langs_to_lre_codes,
                 target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir
    files = glob.glob(str(corpus_dir / "**/*.wav"), recursive=True)
    langs = [(Path(f).parent.parent.parent.parent.stem).lower() for f in files]
    if map_langs_to_lre_codes:
        langs = map_to_lre(langs)
    ids = ["{}-{}".format(l, Path(f).stem) for f, l in zip(files, langs)]
    val = np.array(
        [False if Path(f).stem in buggy_files else True for f in files])
    non_val = np.any(val == False)
    df = pd.DataFrame({"id": ids, "language": langs, "filename": files})
    if non_val:
        df = df.loc[val]
        logging.info("detected invalid files %d / %d remained", len(df),
                     len(val))
    if remove_langs is not None:
        for lang in remove_langs:
            df = df[df["language"] != lang]

    # sort by segment id
    df.sort_values(by=["language", "id"], inplace=True)

    file_lists = []
    file_list = []
    seg_count = 0
    prev_lang = ""
    cat_segs = []
    cur_seg = 0
    for r in range(len(df)):
        row = df.iloc[r]
        if seg_count == 5 or (row.language != prev_lang and seg_count > 0):
            file_lists.append(file_list)
            cat_segs.append(cur_seg)
            file_list = []
            seg_count = 0
            cur_seg = r

        file_list.append(row.filename)
        seg_count += 1
        prev_lang = row.language

    if file_list:
        file_lists.append(file_list)
        cat_segs.append(cur_seg)

    df_cat = df.iloc[cat_segs].drop(["filename"], axis=1)
    df_cat["file_lists"] = file_lists

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
        description=
        "Prepares NCHLT, FLEURS, Lwazi, and AMMI corpus for training")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument("--remove-langs",
                        default=None,
                        nargs="+",
                        help="languages to remove")
    parser.add_argument(
        "--map-langs-to-lre-codes",
        default=False,
        action=ActionYesNo,
        help="use LRE17 language codes",
    )

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
    prepare_data(**namespace_to_dict(args))
