#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger

from enum import Enum


class LangTrialCond(Enum):
    ENG_ENG = 1
    ENG_CMN = 2
    ENG_YUE = 3
    CMN_CMN = 4
    CMN_YUE = 5
    YUE_YUE = 6
    OTHER_OTHER = 7
    OTHER_ENG = 8
    OTHER_CMN = 9
    OTHER_YUE = 10

    @staticmethod
    def is_eng(val):
        if val in "ENG" or val in "USE":
            return True
        return False

    @staticmethod
    def get_side_cond(val):
        if val == "ENG" or val == "USE":
            return "ENG"
        if "YUE" in val:
            return "YUE"
        if "CMN" in val:
            return "CMN"

        return "OTHER"

    @staticmethod
    def get_trial_cond(enr, test):
        enr = LangTrialCond.get_side_cond(enr)
        test = LangTrialCond.get_side_cond(test)
        trial = enr + "_" + test
        try:
            return LangTrialCond[trial]
        except:
            trial = test + "_" + enr
            return LangTrialCond[trial]


class SourceTrialCond(Enum):
    CTS_CTS = 1
    CTS_AFV = 2
    AFV_AFV = 3

    @staticmethod
    def get_trial_cond(enr, test):
        trial = enr.upper() + "_" + test.upper()
        try:
            return SourceTrialCond[trial]
        except:
            trial = test.upper() + "_" + enr.upper()
            return SourceTrialCond[trial]


def write_wav(df, target_fs, wav_dir, output_file):
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            segment_id = row["id"]
            ext = segment_id.split(".")[-1]
            if ext == "flac":
                if target_fs == 16000:
                    wav = f"{wav_dir}/{segment_id}"
                else:
                    wav = f"sox {wav_dir}/{segment_id} -t wav -r {target_fs} - |"
            elif ext == "mp4":
                wav = f"ffmpeg -v 8 -i {wav_dir}/{segment_id} -vn -ar {target_fs} -ac 1 -f wav - |"
            else:
                wav = f"sph2pipe -f wav -p -c 1 {wav_dir}/{segment_id} |"
                if target_fs != 8000:
                    wav = f"{wav} sox -t wav - -t wav -r {target_fs} - |"
            f.write(f"{segment_id} {wav}\n")


def make_enroll_dir(df_segms, wav_dir, target_fs, source, output_path):
    # fix source
    df_segms.loc[df_segms["id"].str.match(r".*\.flac$"), "source_type"] = "afv"
    enroll_dir = Path(output_path + f"_enroll_{source}")
    wav_dir = wav_dir / "enrollment"
    logging.info("making enrollment dir %s", enroll_dir)
    enroll_dir.mkdir(parents=True, exist_ok=True)
    df_segms = (df_segms[(df_segms["partition"] == "enrollment")
                         & (df_segms["source_type"] == source) &
                         (df_segms["language"] != "other")].drop(
                             ["partition"], axis=1).sort_values(by="id"))
    segment_file = enroll_dir / "segments.csv"
    df_segms.to_csv(segment_file, sep=",", index=False)

    with open(enroll_dir / "utt2spk", "w") as f1, open(enroll_dir / "spk2utt",
                                                       "w") as f2:
        for u in df_segms["id"]:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(enroll_dir / "utt2lang", "w") as f:
        for u, s in zip(df_segms["id"], df_segms["language"]):
            f.write(f"{u} {s}\n")

    write_wav(df_segms, target_fs, wav_dir, enroll_dir / "wav.scp")


def make_test_dir(df_segms, wav_dir, target_fs, source, output_path):
    if source == "na":
        # fix source
        df_segms.loc[df_segms["id"].str.match(r".*\.mp4$"),
                     "source_type"] = "afv"
        source = "afv"

    test_dir = Path(output_path + f"_test_{source}")
    wav_dir = wav_dir / "test"
    logging.info("making test dir %s", test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    df_segms = (df_segms[(df_segms["partition"] == "test")
                         & (df_segms["source_type"] == source) &
                         (df_segms["language"] != "other")].drop(
                             ["partition"], axis=1).sort_values(by="id"))

    segment_file = test_dir / "segments.csv"
    df_segms.to_csv(segment_file, sep=",", index=False)

    with open(test_dir / "utt2spk", "w") as f1, open(test_dir / "spk2utt",
                                                     "w") as f2:
        for u in df_segms["id"]:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(test_dir / "utt2lang", "w") as f:
        for u, s in zip(df_segms["id"], df_segms["language"]):
            f.write(f"{u} {s}\n")

    with open(test_dir / "spk2gender", "w") as f:
        for u, g in zip(df_segms["id"], df_segms["gender"]):
            g = g[0]
            f.write(f"{u} {g}\n")

    write_wav(df_segms, target_fs, wav_dir, test_dir / "wav.scp")


def prepare_sre21av_dev_audio(corpus_dir, output_path, av_output_path,
                              target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_path)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir / "data" / "audio"
    segments_file = corpus_dir / "docs" / "sre21_dev_segment_key.tsv"
    df_segms = pd.read_csv(segments_file, sep="\t")
    df_segms.rename(
        columns={
            "segmentid": "id",
            "subjectid": "speaker_id"
        },
        inplace=True,
    )
    df_segms.replace({"language": "english"}, {"language": "eng-zho"},
                     inplace=True)
    df_segms.replace({"language": "cantonese"}, {"language": "zho-yue"},
                     inplace=True)
    df_segms.replace({"language": "mandarin"}, {"language": "zho-cmn"},
                     inplace=True)

    enroll_file = corpus_dir / "docs" / "sre21_audio_dev_enrollment.tsv"

    make_enroll_dir(df_segms, wav_dir, target_fs, "cts", output_path)
    make_enroll_dir(df_segms, wav_dir, target_fs, "afv", output_path)
    make_test_dir(df_segms, wav_dir, target_fs, "cts", output_path)
    make_test_dir(df_segms, wav_dir, target_fs, "afv", output_path)

    wav_dir = corpus_dir / "data" / "video"
    make_test_dir(df_segms, wav_dir, target_fs, "na", av_output_path)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares SRE21 dev audio part")

    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-path",
                        required=True,
                        help="Output data path prefix")
    parser.add_argument(
        "--av-output-path",
        required=True,
        help="Output data path prefix for audio visual",
    )
    parser.add_argument("--target-fs",
                        default=16000,
                        type=int,
                        help="Target sampling frequency")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()
    prepare_sre21av_dev_audio(**namespace_to_dict(args))
