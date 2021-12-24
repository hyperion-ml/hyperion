#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import sys
import os
import argparse
import time
import logging
import subprocess
import re

import numpy as np
import pandas as pd


def find_audios(wav_path):

    command = 'find %s -name "*.flac"' % (wav_path)
    wavs = subprocess.check_output(command, shell=True).decode("utf-8").splitlines()
    keys = [os.path.splitext(os.path.basename(wav))[0] for wav in wavs]
    data = {"key": keys, "file_path": wavs}
    df_wav = pd.DataFrame(data)
    return df_wav


def rttm_is_sorted_by_tbeg(rttm):
    tbeg = rttm["tbeg"].values
    file_id = rttm["file_id"].values
    return np.all(np.logical_or(tbeg[1:] - tbeg[:-1] >= 0, file_id[1:] != file_id[:-1]))


def sort_rttm(rttm):
    return rttm.sort_values(by=["file_id", "tbeg"])


def read_rttm(rttm_file, uem_file=None, sep=" "):

    rttm = pd.read_csv(
        rttm_file,
        sep=sep,
        header=None,
        names=[
            "segment_type",
            "file_id",
            "chnl",
            "tbeg",
            "tdur",
            "ortho",
            "stype",
            "name",
            "conf",
            "slat",
        ],
    )
    # remove empty lines:
    index = rttm["tdur"] >= 0.025
    rttm = rttm[index]
    rttm["ortho"] = "<NA>"
    rttm["stype"] = "<NA>"
    if not rttm_is_sorted_by_tbeg(rttm):
        print("RTTM %s not properly sorted, sorting it" % (rttm_file))
        rttm = sort_rttm(rttm)

    # cross with uem
    if uem_file is not None:
        uem = pd.read_csv(
            uem_file,
            sep=" ",
            header=None,
            names=["file_id", "chnl", "file_tbeg", "file_tend"],
        )
        rttm_uem = pd.merge(left=rttm, right=uem, on=["file_id", "chnl"])

        # fix exceding file duration
        index_fix = (rttm_uem["tbeg"] < rttm_uem["file_tend"]) & (
            rttm_uem["tbeg"] + rttm_uem["tdur"] > rttm_uem["file_tend"]
        )
        if np.sum(index_fix) > 0:
            print(
                "fixing %d segments with exceding file duration" % (np.sum(index_fix))
            )
            # print(rttm_uem[index_fix])
            rttm_uem.loc[index_fix, "tdur"] = (
                rttm_uem[index_fix].file_tend - rttm_uem[index_fix].tbeg
            )

        index_keep = rttm_uem["tbeg"] < rttm_uem["file_tend"]
        n_rm = rttm_uem.shape[0] - np.sum(index_keep)
        if n_rm > 0:
            print("removing %d segments that start after file tend" % (n_rm))
            # print(rttm_uem[~index_keep])
            rttm_uem = rttm_uem[index_keep]

        index_fix = (rttm_uem["tbeg"] < rttm_uem["file_tbeg"]) & (
            rttm_uem["tbeg"] + rttm_uem["tdur"] > rttm_uem["file_tbeg"]
        )
        if np.sum(index_fix) > 0:
            print(
                "fixing %d segments that start before file tbeg" % (np.sum(index_fix))
            )
            # print(rttm_uem[index_fix])
            rttm_uem.loc[index_fix, "tdur"] = (
                rttm_uem[index_fix].tbeg
                + rttm_uem[index_fix].tdur
                - rttm_uem[index_fix].file_tbeg
            )
            rttm_uem.loc[index_fix, "tbeg"] = rttm_uem[index_fix].file_tbeg

        index_keep = rttm_uem["tbeg"] + rttm_uem["tdur"] > rttm_uem["file_tbeg"]
        n_rm = rttm_uem.shape[0] - np.sum(index_keep)
        if n_rm > 0:
            print("removing %d segments that end before tbeg" % (n_rm))
            # print(rttm_uem[~index_keep])
            rttm_uem = rttm_uem[index_keep]

        rttm = rttm_uem.drop(columns=["file_tbeg", "file_tend"])
    # print(pd.concat([rttm,rttm2]).drop_duplicates(keep=False).to_string())

    return rttm


def make_train_segments_from_rttm(df_rttm, min_dur, max_dur):

    segments = pd.DataFrame()
    # vad = pd.DataFrame()
    vad = []
    rng = np.random.RandomState(seed=1234)
    spk_ids = df_rttm["name"].sort_values().unique()
    for spk_id in spk_ids:
        print("make train segments for spk=%s" % (spk_id))
        index = df_rttm["name"] == spk_id
        df_rttm_i = df_rttm[index]
        file_names = df_rttm_i["file_id"].sort_values().unique()
        for file_name in file_names:
            print("\tmake train segments for spk=%s file_name=%s" % (spk_id, file_name))
            index = df_rttm_i["file_id"] == file_name
            df_rttm_ij = df_rttm_i[index]
            cum_length = np.cumsum(np.asarray(df_rttm_ij["tdur"]))
            total_length = cum_length[-1]
            first_utt = 0
            count = 0
            while total_length > min_dur:
                # select number of utterances for this segment
                cur_dur = min(rng.randint(min_dur, max_dur), total_length)
                # print('\t\t extract segment %d of length %.2f, remaining length %.2f' % (count, cur_dur, total_length-cur_dur))
                last_utt = np.where(cum_length >= cur_dur)[0][0]
                tbeg = df_rttm_ij.iloc[first_utt].tbeg - 1
                tbeg = tbeg if tbeg > 0 else 0
                tend = df_rttm_ij.iloc[last_utt].tbeg + df_rttm_ij.iloc[last_utt].tdur

                # make segment
                segment_id = "%s-%s-%07d-%07d" % (
                    spk_id,
                    file_name,
                    int(tbeg * 100),
                    int(tend * 100),
                )
                row = {
                    "segment_id": segment_id,
                    "filename": file_name,
                    "speaker": spk_id,
                    "beginning_time": tbeg,
                    "end_time": tend,
                }
                segments = segments.append(row, ignore_index=True)

                # make vad
                df_vad = df_rttm_ij.iloc[first_utt : last_utt + 1].copy()
                df_vad["file_id"] = segment_id
                df_vad["name"] = "speech"
                df_vad["tbeg"] = df_vad["tbeg"] - tbeg
                vad.append(df_vad)
                # vad = pd.concat([vad, df_vad], ignore_index=True)

                # update remaining length for current speaker in current audio
                cum_length -= cum_length[last_utt]
                total_length = cum_length[-1]
                first_utt = last_utt + 1
                count += 1

    vad = pd.concat(vad, ignore_index=True)
    segments.sort_values("segment_id", inplace=True)
    vad.sort_values(["file_id", "tbeg"], inplace=True)

    return segments, vad


def segm_vad_to_rttm_vad(segments):

    file_id = segments.segment_id
    tbeg = segments.beginning_time
    tdur = segments.end_time - segments.beginning_time
    num_segments = len(file_id)
    segment_type = ["SPEAKER"] * num_segments

    nans = ["<NA>" for i in range(num_segments)]
    chnl = [1 for i in range(num_segments)]
    ortho = nans
    stype = nans
    name = segments.speaker
    conf = [1 for i in range(num_segments)]
    slat = nans

    df = pd.DataFrame(
        {
            "segment_type": segment_type,
            "file_id": file_id,
            "chnl": chnl,
            "tbeg": tbeg,
            "tdur": tdur,
            "ortho": ortho,
            "stype": stype,
            "name": name,
            "conf": conf,
            "slat": slat,
        }
    )
    df["name"] = "speech"
    return df


def remove_overlap_from_rttm_vad(rttm):

    tbeg_index = rttm.columns.get_indexer(["tbeg"])
    tdur_index = rttm.columns.get_indexer(["tdur"])
    tend = np.asarray(rttm["tbeg"] + rttm["tdur"])
    index = np.ones(rttm.shape[0], dtype=bool)
    p = 0
    for i in range(1, rttm.shape[0]):
        if rttm["file_id"].iloc[p] == rttm["file_id"].iloc[i]:
            if tend[p] > rttm.iloc[i, tbeg_index].item():
                index[i] = False
                if tend[i] > tend[p]:
                    tend[p] = tend[i]
                    new_dur = tend[i] - rttm.iloc[p, tbeg_index].item()
                    rttm.iloc[p, tdur_index] = new_dur
            else:
                p = i
        else:
            p = i

    rttm = rttm.loc[index]
    return rttm


def filter_wavs(df_wav, file_names):
    df_wav = df_wav.loc[df_wav["key"].isin(file_names)].sort_values("key")
    return df_wav


def write_wav(df_wav, df_segments, output_path):

    df_wav.index = df_wav.key
    with open(output_path + "/wav.scp", "w") as f:
        for segment_id, file_id, tbeg, tend in zip(
            df_segments["segment_id"],
            df_segments["filename"],
            df_segments["beginning_time"],
            df_segments["end_time"],
        ):
            file_path = df_wav.loc[file_id, "file_path"]
            f.write(
                "%s sox -t flac %s -t wav - trim %.3f =%.3f | \n"
                % (segment_id, file_path, tbeg, tend)
            )


def write_utt2spk_from_segm(df_seg, output_path):

    with open(output_path + "/utt2spk", "w") as f:
        for u, s in zip(df_seg["segment_id"], df_seg["speaker"]):
            f.write("%s %s\n" % (u, s))


def write_dummy_utt2spk(file_names, output_path):

    with open(output_path + "/utt2spk", "w") as f:
        for fn in file_names:
            f.write("%s %s\n" % (fn, fn))


def write_segments(df_seg, output_path):

    with open(output_path + "/segments", "w") as f:
        for i, row in df_seg.iterrows():
            f.write(
                "%s %s %.2f %.2f\n"
                % (
                    row["segment_id"],
                    row["filename"],
                    row["beginning_time"],
                    row["end_time"],
                )
            )


def write_rttm_vad(df_vad, output_path):
    file_path = output_path + "/vad.rttm"
    df_vad[
        [
            "segment_type",
            "file_id",
            "chnl",
            "tbeg",
            "tdur",
            "ortho",
            "stype",
            "name",
            "conf",
            "slat",
        ]
    ].to_csv(file_path, sep=" ", float_format="%.3f", index=False, header=False)


def write_rttm_spk(df_vad, output_path):
    file_path = output_path + "/diarization.rttm"
    df_vad[
        [
            "segment_type",
            "file_id",
            "chnl",
            "tbeg",
            "tdur",
            "ortho",
            "stype",
            "name",
            "conf",
            "slat",
        ]
    ].to_csv(file_path, sep=" ", float_format="%.3f", index=False, header=False)


def make_train(df_wav, df_rttm, output_path, min_dur, max_dur):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # make train segments and vad
    print("make training segments")
    segments, vad = make_train_segments_from_rttm(df_rttm, min_dur, max_dur)
    print("write utt2spk")
    write_utt2spk_from_segm(segments, output_path)

    # create wav.scp
    print("make wav.scp")
    write_wav(df_wav, segments, output_path)

    # print('write segments')
    # write_segments(segments, output_path)
    print("write vad in rttm format")
    write_rttm_vad(vad, output_path)


def make_dihard_train(
    rttm_file, uem_file, wav_path, output_path, min_dur, max_dur, data_prefix
):

    print("read audios")
    df_wav = find_audios(wav_path)
    print("read rttm")
    rttm = read_rttm(rttm_file, uem_file)
    rttm["name"] = data_prefix + rttm["name"].astype(str)

    print("making data directory %s" % (output_path))
    make_train(df_wav, rttm, output_path, min_dur, max_dur)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Prepares DIHARD data for PLDA/x-vector training",
    )

    parser.add_argument("--rttm", dest="rttm_file", required=True)
    parser.add_argument("--uem", dest="uem_file", default=None)
    parser.add_argument("--wav-path", dest="wav_path", required=True)
    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--data-prefix", dest="data_prefix", required=True)
    parser.add_argument("--min-train-dur", dest="min_dur", default=15, type=float)
    parser.add_argument("--max-train-dur", dest="max_dur", default=60, type=float)

    args = parser.parse_args()

    make_dihard_train(**vars(args))
