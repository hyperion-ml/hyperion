#!/usr/bin/env python
"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import List, Union

import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.metrics import VoxProfileEvaluator as VPE
from hyperion.utils.misc import PathLike


def accum_voxprofile_metrics(
    input_files: Union[List[PathLike], None],
    output_file: PathLike,
    num_parts: int,
    base_idx: int,
):
    """Aggregate VoxProfile metric tables and write merged statistics.

    Args:
        input_files: Optional list of metric files to merge. If ``None``, file
            names are derived from ``output_file`` using ``num_parts`` and
            ``base_idx``.
        output_file: Destination CSV/TSV file for the accumulated metrics.
        num_parts: Number of part files to derive when ``input_files`` is
            ``None``.
        base_idx: Starting index used when deriving part-file names.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    assert input_files is not None or num_parts != 0
    ext = output_file.suffix

    if input_files is None:
        input_file_base = output_file.with_suffix("")
        input_files = []
        logging.info(
            "accumulating %s* -> %s", input_file_base.with_suffix(""), output_file
        )
        for i in range(num_parts):
            idx = base_idx + i
            input_file_i = input_file_base.with_suffix(f".{idx}{ext}")
            input_files.append(input_file_i)
    else:
        logging.info("merging %s -> %s", " + ".join(input_files), output_file)

    sep = "\t" if ext == ".tsv" else ","
    tables = []
    for file_path in input_files:
        table_i = pd.read_csv(file_path, sep=sep)
        tables.append(table_i)

    output_table = VPE.accum_stats(tables)
    output_table.to_csv(output_file, sep=sep, index=False, float_format="{:.4f}".format)

    pd.options.display.float_format = "{:.4}".format
    print(output_table.to_string(), flush=True)


def main():
    """Parse CLI arguments and run VoxProfile metric accumulation."""
    parser = ArgumentParser(
        description=(
            "Tool to accumulate VoxProfile metrics obtained by different subprocesses"
        )
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--input-files",
        default=None,
        nargs="+",
        help="optional list of VoxProfile metric files to merge",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help=(
            "output CSV/TSV file; if --input-files is not provided, input part-file "
            "names are derived from this path"
        ),
    )
    parser.add_argument(
        "--num-parts",
        default=0,
        type=int,
        help="number of part files to merge when --input-files is not provided",
    )

    parser.add_argument(
        "--base-idx",
        default=1,
        type=int,
        help="starting part index used to derive input file names (typically 0 or 1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )

    args = parser.parse_args()
    kwargs = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    accum_voxprofile_metrics(**kwargs)


if __name__ == "__main__":
    main()
