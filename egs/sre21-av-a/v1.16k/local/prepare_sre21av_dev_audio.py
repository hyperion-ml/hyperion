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
            segment_id = row["segment_id"]
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


def make_enroll_dir(df_segms, df_enr, wav_dir, target_fs, output_path):
    enroll_dir = Path(output_path + "_enroll")
    wav_dir = wav_dir / "enrollment"
    logging.info("making enrollment dir %s", enroll_dir)
    enroll_dir.mkdir(parents=True, exist_ok=True)
    df_segms = (
        df_segms[
            (df_segms["partition"] == "enrollment")
            & df_segms["source_type"].isin(["cts", "afv"])
        ]
        .drop(["partition"], axis=1)
        .sort_values(by="segment_id")
    )
    segment_file = enroll_dir / "segments.csv"
    df_segms.to_csv(segment_file, sep=",", index=False)

    with open(enroll_dir / "utt2spk", "w") as f1, open(
        enroll_dir / "spk2utt", "w"
    ) as f2:
        for u in df_segms["segment_id"]:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(enroll_dir / "utt2lang", "w") as f:
        for u, s in zip(df_segms["segment_id"], df_segms["language"]):
            f.write(f"{u} {s}\n")

    with open(enroll_dir / "utt2model", "w") as f:
        for u, s in zip(df_enr["segment_id"], df_enr["model_id"]):
            f.write(f"{u} {s}\n")

    write_wav(df_segms, target_fs, wav_dir, enroll_dir / "wav.scp")

    # fix source
    df_segms.loc[df_segms["segment_id"].str.match(r".*\.flac$"), "source_type"] = "afv"

    df_merge = pd.merge(df_segms, df_enr, on="segment_id")
    df_model = (
        df_merge[["model_id", "gender", "language", "source_type"]]
        .drop_duplicates()
        .sort_values(by="model_id")
    )

    df_model.to_csv(enroll_dir / "model.csv", sep=",", index=False)
    return df_model


def write_simple_trialfile(df_key, output_file):
    df_key.to_csv(
        output_file,
        sep=" ",
        columns=["model_id", "segment_id", "targettype"],
        index=False,
        header=False,
    )


def make_test_dir(df_segms, df_model, df_key, wav_dir, target_fs, sources, output_path):
    test_dir = Path(output_path + "_test")
    wav_dir = wav_dir / "test"
    logging.info("making test dir %s", test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    df_segms = (
        df_segms[
            (df_segms["partition"] == "test") & df_segms["source_type"].isin(sources)
        ]
        .drop(["partition"], axis=1)
        .sort_values(by="segment_id")
    )

    if sources[0] == "na":
        # fix source
        df_segms.loc[
            df_segms["segment_id"].str.match(r".*\.mp4$"), "source_type"
        ] = "afv"

    segment_file = test_dir / "segments.csv"
    df_segms.to_csv(segment_file, sep=",", index=False)

    with open(test_dir / "utt2spk", "w") as f1, open(test_dir / "spk2utt", "w") as f2:
        for u in df_segms["segment_id"]:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(test_dir / "utt2lang", "w") as f:
        for u, s in zip(df_segms["segment_id"], df_segms["language"]):
            f.write(f"{u} {s}\n")

    with open(test_dir / "spk2gender", "w") as f:
        for u, g in zip(df_segms["segment_id"], df_segms["gender"]):
            g = g[0]
            f.write(f"{u} {g}\n")

    write_wav(df_segms, target_fs, wav_dir, test_dir / "wav.scp")

    df_model.index = df_model.model_id
    df_segms.index = df_segms.segment_id
    last_model = ""
    # make trials
    lang_cond = [None] * len(df_key)
    source_cond = [None] * len(df_key)
    for i in range(len(df_key)):
        row = df_key.iloc[i]
        if row.model_id != last_model:
            model = df_model.loc[row.model_id]

        test = df_segms.loc[row.segment_id]

        lang_cond[i] = LangTrialCond.get_trial_cond(
            model["language"], test["language"]
        ).name
        source_cond[i] = SourceTrialCond.get_trial_cond(
            model["source_type"], test["source_type"]
        ).name

    df_key["source_type"] = source_cond
    df_key["lang_cond"] = lang_cond

    df_key.to_csv(test_dir / "trials.csv", sep=",", index=False)
    write_simple_trialfile(df_key, test_dir / "trials")
    for cond in SourceTrialCond:
        df_cond = df_key[df_key["source_type"] == cond.name]
        write_simple_trialfile(df_cond, test_dir / f"trials_{cond.name}")

    for cond in LangTrialCond:
        df_cond = df_key[df_key["lang_cond"] == cond.name]
        write_simple_trialfile(df_cond, test_dir / f"trials_{cond.name}")

    for nenr in [1, 3]:
        df_cond = df_key[df_key["num_enroll_segs"] == nenr]
        write_simple_trialfile(df_cond, test_dir / f"trials_nenr{nenr}")

    for cond_name, cond in zip(["samephn", "diffphn"], ["Y", "N"]):
        df_cond = df_key[
            (df_key["phone_num_match"] == cond) | (df_key["targettype"] == "nontarget")
        ]
        write_simple_trialfile(df_cond, test_dir / f"trials_{cond_name}")

    for cond_name, cond in zip(["samelang", "difflang"], ["Y", "N"]):
        df_cond = df_key[df_key["language_match"] == cond]
        write_simple_trialfile(df_cond, test_dir / f"trials_{cond_name}")

    for cond_name, cond in zip(["samesource", "diffsource"], ["Y", "N"]):
        df_cond = df_key[df_key["source_type_match"] == cond]
        write_simple_trialfile(df_cond, test_dir / f"trials_{cond_name}")

    for cond in ["male", "female"]:
        df_cond = df_key[df_key["gender"] == cond]
        write_simple_trialfile(df_cond, test_dir / f"trials_{cond}")


def prepare_sre21av_dev_audio(
    corpus_dir, output_path, av_output_path, target_fs, verbose
):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_path)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir / "data" / "audio"
    segments_file = corpus_dir / "docs" / "sre21_dev_segment_key.tsv"
    df_segms = pd.read_csv(segments_file, sep="\t")
    df_segms.rename(
        columns={"segmentid": "segment_id", "subjectid": "speaker_id"},
        inplace=True,
    )
    df_segms.replace({"language": "english"}, {"language": "ENG"}, inplace=True)
    df_segms.replace({"language": "cantonese"}, {"language": "YUE"}, inplace=True)
    df_segms.replace({"language": "mandarin"}, {"language": "CMN"}, inplace=True)

    enroll_file = corpus_dir / "docs" / "sre21_audio_dev_enrollment.tsv"
    df_enr = pd.read_csv(enroll_file, sep="\t")
    df_enr.rename(
        columns={"segmentid": "segment_id", "modelid": "model_id"},
        inplace=True,
    )
    key_file = corpus_dir / "docs" / "sre21_audio_dev_trial_key.tsv"
    df_key = pd.read_csv(key_file, sep="\t")
    df_key.rename(
        columns={"segmentid": "segment_id", "modelid": "model_id"},
        inplace=True,
    )

    df_model = make_enroll_dir(df_segms, df_enr, wav_dir, target_fs, output_path)
    make_test_dir(
        df_segms, df_model, df_key, wav_dir, target_fs, ["cts", "afv"], output_path
    )

    key_file = corpus_dir / "docs" / "sre21_audio-visual_dev_trial_key.tsv"
    df_key = pd.read_csv(key_file, sep="\t")
    df_key.rename(
        columns={"segmentid": "segment_id", "modelid": "model_id"},
        inplace=True,
    )
    wav_dir = corpus_dir / "data" / "video"
    make_test_dir(
        df_segms, df_model, df_key, wav_dir, target_fs, ["na"], av_output_path
    )


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares SRE21 dev audio part")

    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--output-path", required=True, help="Ouput data path prefix")
    parser.add_argument(
        "--av-output-path",
        required=True,
        help="Ouput data path prefix for audio visual",
    )
    parser.add_argument(
        "--target-fs", default=16000, type=int, help="Target sampling frequency"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_sre21av_dev_audio(**namespace_to_dict(args))
