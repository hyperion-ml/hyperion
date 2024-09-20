#!/usr/bin/env python
"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from tqdm import tqdm

from hyperion.hyp_defs import config_logger
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import VADReaderFactory as VRF
from hyperion.utils import InfoTable, PathLike, RecordingSet, SegmentSet, VADSet

subcommand_list = ["bin_to_time_marks", "time_marks_to_bin"]


def add_common_args(parser):
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--in-vad-file", required=True, help="input VADSet file")
    parser.add_argument("--out-vad-file", required=True, help="output VADSet file")
    parser.add_argument("--path-prefix", default=None, help="prefix for vad paths")

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def make_bin_to_time_marks_parser():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--output-dir", required=True, help="output directory to write table files"
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["tsv", "csv"],
        help="table format in [csv, tsv]",
    )
    return parser


def bin_to_time_marks(
    in_vad_file: PathLike,
    out_vad_file: PathLike,
    output_dir: PathLike,
    path_prefix: Optional[PathLike] = None,
    format: str = "csv",
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, is_ok=True)
    if format == "tsv":
        sep = "\t"
        ext = ".tsv"
    else:
        sep = ","
        ext = ",csv"

    v_reader = VRF.create(in_vad_file, path_prefix=path_prefix)
    ids = v_reader.keys
    output_files = []
    for id in tqdm(ids):
        vad = v_reader.read_time_marks([id])[0]
        output_file = output_dir / Path(id).with_suffix(ext)
        vad.to_csv(output_file, sep=sep, index=False)
        output_files.append(output_file)

    df_vad = pd.DataFrame({"id": ids, "storage_path": output_files})
    vad_set = VADSet(df_vad)
    vad_set.save(out_vad_file)


def make_time_marks_to_bin_parser():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--frame-length",
        default=25.0,
        type=float,
        help="frame length of the binary vad",
    )
    parser.add_argument(
        "--frame-shift", default=10.0, type=float, help="frame shift of the binary vad"
    )
    parser.add_argument(
        "--snip-edges",
        default=False,
        action=ActionYesNo,
        help="snip edges in binary vad",
    )
    parser.add_argument(
        "--segments-file", default=None, help="segments file to get maximum durations"
    )
    return parser


def time_marks_to_bin(
    in_vad_file: PathLike,
    out_vad_file: PathLike,
    path_prefix: Optional[PathLike] = None,
    frame_shift: float = 10.0,
    frame_length: float = 25.0,
    snip_edges: bool = False,
    segments_file: Optional[PathLike] = None,
):

    v_reader = VRF.create(in_vad_file, path_prefix=path_prefix)
    segments = SegmentSet.load(segments_file)
    ids = v_reader.keys
    metadata_columns = [
        "frame_shift",
        "frame_length",
        "num_frames",
        "num_speech_frames",
        "prob_speech",
    ]

    with DWF.create(out_vad_file, metadata_columns=metadata_columns) as writer:
        for id in tqdm(ids):
            if segments is not None:
                duration = segments.loc[id, "duration"]
            else:
                duration = None

            vad = v_reader.read_binary(
                [id],
                frame_shift=frame_shift,
                frame_length=frame_length,
                snip_edges=snip_edges,
                duration=duration,
            )[0]
            metadata = {
                "frame_shift": frame_shift,
                "frame_length": frame_length,
                "num_frames": vad.shape[0],
                "num_speech_frames": np.sum(vad),
                "prob_speech": np.sum(vad) / vad.shape[0],
            }
            writer.write([id], [vad], metadata)


def main():
    parser = ArgumentParser(description="Tool to manipulates the Hyperion data tables")
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for subcommand in subcommand_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()
