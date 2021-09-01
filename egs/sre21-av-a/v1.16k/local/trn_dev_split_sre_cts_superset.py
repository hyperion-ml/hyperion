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
from hyperion.utils import SCPList, TrialKey
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


def split_segments_into_trn_dev(df, num_dev_spks_cmn, num_dev_spks_yue):
    """Splits segments into train and dev
    it leaves the same number of male and feamle speakers
    for development.
    """
    # find utterances containing CMN or YUE
    idx_cmn = df.language == "CMN"
    idx_yue = df.language == "YUE"
    idx_male = df.gender == "male"
    idx_female = df.gender == "female"
    # find speakers
    spks_cmn_male = df[idx_cmn & idx_male].speaker_id.unique()
    spks_cmn_female = df[idx_cmn & idx_female].speaker_id.unique()
    spks_yue_male = df[idx_yue & idx_male].speaker_id.unique()
    spks_yue_female = df[idx_yue & idx_female].speaker_id.unique()
    logging.info(
        "Found %d/%d male/female speakers of CMN",
        len(spks_cmn_male),
        len(spks_cmn_female),
    )
    logging.info(
        "Found %d/%d male/female speakers of YUE",
        len(spks_yue_male),
        len(spks_yue_female),
    )

    logging.info("Selecting %d CMN first speakers for dev", num_dev_spks_cmn)
    spks_cmn_male = spks_cmn_male[: num_dev_spks_cmn // 2]
    spks_cmn_female = spks_cmn_female[: num_dev_spks_cmn // 2]
    logging.info("Selecting %d YUE first speakers for dev", num_dev_spks_yue)
    spks_yue_male = spks_yue_male[: num_dev_spks_yue // 2]
    spks_yue_female = spks_yue_female[: num_dev_spks_yue // 2]
    spks_dev = np.unique(
        np.concatenate((spks_cmn_male, spks_cmn_female, spks_yue_male, spks_yue_female))
    )
    logging.info("Selected %d dev speakers", len(spks_dev))

    df_trn = df[~df["speaker_id"].isin(spks_dev)]
    df_dev = df[df["speaker_id"].isin(spks_dev)]
    logging.info(
        "Train-segments: %d/%d, Dev-segments %d/%d",
        len(df_trn),
        len(df),
        len(df_dev),
        len(df),
    )
    return df_trn, df_dev


def make_data_dir(df, input_dir, output_dir):
    """Creates the Kaldi data directory from the selected segments table"""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("saving segments.csv")
    segments_file = output_dir / "segments.csv"
    df.to_csv(segments_file, sep=",", index=False)

    # Kaldi data directory files
    # utt2xxx files
    logging.info("saving Kaldi utt2xxx files")
    columns = [
        "speaker_id",
        "speech_duration",
        "session_id",
        "corpus_id",
        "phone_id",
        "language",
    ]
    files = [
        "utt2spk",
        "utt2speech_dur",
        "utt2session",
        "utt2corpus",
        "utt2phone",
        "utt2lang",
    ]
    for c, f in zip(columns, files):
        output_file = output_dir / f
        df.to_csv(
            output_file, sep=" ", columns=["segment_id", c], header=False, index=False
        )

    # speaker table
    logging.info("saving speaker files")
    spk_df = df[["speaker_id", "gender"]].drop_duplicates()
    output_file = output_dir / "speaker.csv"
    spk_df.to_csv(output_file, sep=",", index=False)
    gender = df["gender"].str.replace("female", "f").str.replace("male", "m")
    spk_df["gender"] = gender
    output_file = output_dir / "spk2gender"
    spk_df.to_csv(output_file, sep=" ", header=False, index=False)

    with open(output_dir / "spk2utt", "w") as f:
        for spk in df["speaker_id"].unique():
            segment_ids = " ".join(df[df["speaker_id"] == spk].segment_id.values)
            f.write(f"{spk} {segment_ids}\n")

    # wav.scp
    logging.info("saving wav.scp")
    in_wav = input_dir / "wav.scp"
    out_wav = output_dir / "wav.scp"
    scp = SCPList.load(in_wav)
    scp = scp.filter(df.segment_id.values)
    scp.save(out_wav)


def make_enr_row(segms, spk, gender, lang, phn, sess, count):
    num_enrs = len(segms)
    model_id = "%s-%s-%s-%s-%s-%ds-%03d" % (
        spk,
        gender[0],
        lang,
        phn,
        sess,
        num_enrs,
        count,
    )
    row = {
        "model_id": model_id,
        "segment_ids": segms,
        "speaker_id": spk,
        "gender": gender,
        "language": lang,
        "phone_id": phn,
        "session_id": sess,
        "num_segms": num_enrs,
    }
    return row


def make_test_row(segm, spk, gender, lang, phn, sess):
    row = {
        "segment_id": segm,
        "speaker_id": spk,
        "gender": gender,
        "language": lang,
        "phone_id": phn,
        "session_id": sess,
    }
    return row


def split_session(df, df_enr, df_test):
    """Split segments in a sesssion into enrollment and test

    We are going to make sure that when having 3 segments
    enrollments the 3 segments come from the same session phone number
    and language
    For each tuple session 3 out of 4 segments will
    be asigned to enroll and 1 to test,
    if there are only 2-3 segments, one will be for enroll
    and 1-2 for test
    if there is only 1, it will be for test

    Arguments:
       df: DataFrame with the information of all the segments
       df_enr: Dataframe where we append the info for the enrollemnt models
       df_test: Dataframe where we append the info for the test segments
    """
    segms = df["segment_id"].values
    row = df.iloc[0]
    spk, gender, lang, phn, sess = (
        row["speaker_id"],
        row["gender"],
        row["language"],
        row["phone_id"],
        row["session_id"],
    )
    num_segms = len(segms)
    num_enroll_3s = num_segms // 4
    count = 0
    count_enr1s = 0
    for count_enr3s in range(num_enroll_3s):
        segms_3s = []
        for i in range(3):
            # add segment to enrollment with 1 segment
            cur_segm = segms[count]
            count += 1
            count_enr1s += 1
            segms_3s.append(cur_segm)
            row = make_enr_row([cur_segm], spk, gender, lang, phn, sess, count_enr1s)
            df_enr = df_enr.append(row, ignore_index=True)

        # add the three last segments to enrollment with 3 segments
        row = make_enr_row(segms_3s, spk, gender, lang, phn, sess, count_enr3s)
        df_enr = df_enr.append(row, ignore_index=True)

        # add next segment to test side
        cur_segm = segms[count]
        count += 1
        row = make_test_row(cur_segm, spk, gender, lang, phn, sess)
        df_test = df_test.append(row, ignore_index=True)

    if count == num_segms - 3:
        # if there are 3 segments remaining we add one to enroll
        cur_segm = segms[count]
        count += 1
        count_enr1s += 1
        row = make_enr_row([cur_segm], spk, gender, lang, phn, sess, count_enr1s)
        df_enr = df_enr.append(row, ignore_index=True)

    while count < num_segms:
        # we add all the rest segments to test
        cur_segm = segms[count]
        count += 1
        row = make_test_row(cur_segm, spk, gender, lang, phn, sess)
        df_test = df_test.append(row, ignore_index=True)

    return df_enr, df_test


def split_enr_test(df):
    """Split segments into enrollment and test

    First we split segments into enrollment and test
    We are going to make sure that when having 3 segments
    enrollments the 3 segments come from the same session phone number
    and language
    For each tuple spk-language-phone-session 3 out of 4 segments will
    be asigned to enroll and 1 to test,
    if there are only 2-3 segments, one will be for enroll
    and 1-2 for test
    if there is only 1, it will be for test

    Arguments:
       df: DataFrame with the information of all the segments

    Returns:
       df_enr: Dataframe with the information of the enrollemnt models
       df_test: Dataframe with the information of the test segments
    """
    df_enr = pd.DataFrame(
        columns=[
            "model_id",
            "segment_ids",
            "speaker_id",
            "gender",
            "language",
            "phone_id",
            "session_id",
            "num_segms",
        ]
    )
    df_test = pd.DataFrame(
        columns=[
            "segment_id",
            "speaker_id",
            "gender",
            "language",
            "phone_id",
            "session_id",
        ]
    )

    for spk in df.speaker_id.unique():
        df_spk = df[df.speaker_id == spk]
        for lang in df_spk.language.unique():
            df_sl = df_spk[df_spk.language == lang]
            for phn in df_sl.phone_id.unique():
                df_slp = df_sl[df_sl.phone_id == phn]
                for sess in df_slp.session_id.unique():
                    df_slps = df_slp[df_slp.session_id == sess]
                    df_enr, df_test = split_session(df_slps, df_enr, df_test)

    return df_enr, df_test


def make_trial_key(df_enr, df_test):
    model_set = df_enr["model_id"].values
    seg_set = df_test["segment_id"].values
    key = TrialKey(model_set, seg_set)
    df_enr.index = df_enr.model_id
    df_test.index = df_test.segment_id
    samephn = np.zeros_like(key.tar)
    diffphn = np.zeros_like(key.tar)
    lang = np.zeros_like(key.tar, dtype=np.int32)
    nenr = np.zeros_like(key.tar, dtype=np.int32)
    female = np.zeros_like(key.tar, dtype=np.int32)
    for i, model_id in enumerate(model_set):
        enr_row = df_enr.loc[model_id]
        for j, segment_id in enumerate(seg_set):
            test_row = df_test.loc[segment_id]
            # we only make trials between same gender and different session
            if (
                enr_row["session_id"] != test_row["session_id"]
                and enr_row["gender"] == test_row["gender"]
            ):
                if enr_row["speaker_id"] == test_row["speaker_id"]:
                    key.tar[i, j] = True
                    if enr_row["phone_id"] == test_row["phone_id"]:
                        samephn[i, j] = True
                    else:
                        diffphn[i, j] = True
                else:
                    key.non[i, j] = True
                    samephn[i, j] = True
                    diffphn[i, j] = True

                lang[i, j] = LangTrialCond.get_trial_cond(
                    enr_row["language"], test_row["language"]
                ).value
                nenr[i, j] = enr_row["num_segms"]
                female[i, j] = enr_row["gender"] == "female"

    return key, samephn, diffphn, lang, nenr, female


def make_trials(df, output_dir):
    """Creates enrollment and trial files"""
    df_enr, df_test = split_enr_test(df)
    df_enr.to_csv(output_dir / "enroll.csv", sep=",", index=False)
    df_test.to_csv(output_dir / "test.csv", sep=",", index=False)

    with open(output_dir / "utt2enroll", "w") as f:
        for _, row in df_enr.iterrows():
            model_id = row["model_id"]
            for segm in row["segment_ids"]:
                f.write(f"{segm} {model_id}\n")

    key, samephn, diffphn, lang, nenr, female = make_trial_key(df_enr, df_test)
    key.save(output_dir / "trials")

    # same phone trials
    key_cond = key.copy()
    key_cond.tar = np.logical_and(key.tar, samephn)
    key_cond.non = np.logical_and(key.non, samephn)
    key_cond.save(output_dir / "trials_samephn")

    # diff phone trials
    key_cond = key.copy()
    key_cond.tar = np.logical_and(key.tar, diffphn)
    key_cond.non = np.logical_and(key.non, diffphn)
    key_cond.save(output_dir / "trials_diffphn")

    # language conditions
    for e in LangTrialCond:
        key_cond = key.copy()
        key_cond.tar = np.logical_and(key.tar, lang == e.value)
        key_cond.non = np.logical_and(key.non, lang == e.value)
        key_cond.save(output_dir / f"trials_{e.name}")

    # enrollment conditions
    for n in (1, 3):
        key_cond = key.copy()
        key_cond.tar = np.logical_and(key.tar, nenr == n)
        key_cond.non = np.logical_and(key.non, nenr == n)
        key_cond.save(output_dir / f"trials_nenr{n}")

    key_cond = key.copy()
    key_cond.tar = np.logical_and(key.tar, np.logical_not(female))
    key_cond.non = np.logical_and(key.non, np.logical_not(female))
    key_cond.save(output_dir / "trials_male")

    key_cond = key.copy()
    key_cond.tar = np.logical_and(key.tar, female)
    key_cond.non = np.logical_and(key.non, female)
    key_cond.save(output_dir / "trials_female")


def trn_dev_split_sre_superset(
    input_dir, trn_dir, dev_dir, num_dev_spks_cmn, num_dev_spks_yue, verbose
):

    config_logger(verbose)
    logging.info("Split SRE Superset into Train and Dev for SRE21")
    input_dir = Path(input_dir)
    trn_dir = Path(trn_dir)
    dev_dir = Path(dev_dir)

    df_filepath = input_dir / "segments.csv"
    df = pd.read_csv(df_filepath)
    df_trn, df_dev = split_segments_into_trn_dev(df, num_dev_spks_cmn, num_dev_spks_yue)
    logging.info("Making train dir: %s", trn_dir)
    make_data_dir(df_trn, input_dir, trn_dir)
    logging.info("Making dev dir: %s", dev_dir)
    make_data_dir(df_dev, input_dir, dev_dir)
    logging.info("Making dev trials")
    make_trials(df_dev, dev_dir)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Split SRE Superset into Train and Dev for SRE21"
    )

    parser.add_argument(
        "--input-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--trn-dir", required=True, help="Train data path")
    parser.add_argument("--dev-dir", required=True, help="Dev data path")
    parser.add_argument(
        "--num-dev-spks-cmn",
        required=True,
        type=int,
        help="Number of Mandarin speakers for dev",
    )
    parser.add_argument(
        "--num-dev-spks-yue",
        required=True,
        type=int,
        help="Number of Cantonese speakers for dev",
    )

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    trn_dev_split_sre_superset(**namespace_to_dict(args))
