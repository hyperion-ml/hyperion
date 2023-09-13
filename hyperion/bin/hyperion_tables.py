#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    ClassInfo,
    EnrollmentMap,
    FeatureSet,
    InfoTable,
    PathLike,
    RecordingSet,
    SegmentSet,
)

subcommand_list = ["cat"]
table_dict = {
    "segments": SegmentSet,
    "recordings": RecordingSet,
    "features": FeatureSet,
    "classes": ClassInfo,
    "enrollments": EnrollmentMap,
    "generic": InfoTable,
}


def add_common_args(parser):
    parser.add_argument(
        "--table-type",
        default="generic",
        choices=list(table_dict.keys()),
        help=f"Type of table in {list(table_dict.keys())}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def make_cat_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--input-files", default=None, nargs="+", help="optional list of input files"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="""output file, if input-files is None, input files names are derived from it""",
    )
    parser.add_argument(
        "--num-tables",
        default=0,
        type=int,
        help="""number of jobs we used to create the individual tables""",
    )
    parser.add_argument(
        "--base-idx",
        default=1,
        type=int,
        help="""index of the first job, typically 0 or 1""",
    )

    add_common_args(parser)
    return parser


def cat(
    table_type: str,
    input_files: Union[List[PathLike], None],
    output_file: PathLike,
    num_tables: int,
    base_idx: int = 1,
):
    assert input_files is not None or num_tables != 0
    output_file = Path(output_file)
    if input_files is None:
        ext = output_file.suffix
        input_file_base = output_file.with_suffix("")
        input_files = []
        for i in range(num_tables):
            idx = base_idx + i
            input_file_i = input_file_base.with_suffix(f".{idx}{ext}")
            input_files.append(input_file_i)

    table_class = table_dict[table_type]
    tables = []
    for file_path in input_files:
        tables.append(table_class.load(file_path))

    output_table = table_class.cat(tables)
    output_table.save(output_file)


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
