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
from hyperion.metrics import SpeechQualityEvaluator as SQE
from hyperion.utils.misc import PathLike


def accum_speech_quality_metrics(
    input_files: Union[List[PathLike], None],
    output_file: PathLike,
    num_parts: int,
    base_idx: int,
):
    """Aggregate speech-quality metric tables and write merged statistics.

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

    output_table = SQE.accum_stats(tables)
    output_table.to_csv(output_file, sep=sep, index=False, float_format="{:.4f}".format)

    pd.options.display.float_format = "{:.4}".format
    print(output_table.to_string(), flush=True)


def main():
    """Parse CLI arguments and run speech-quality metric accumulation."""
    parser = ArgumentParser(
        description="Tool to accumulate speech quality metrics obtained by different subprocesses"
    )
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
        "--num-parts",
        default=0,
        type=int,
        help="""number of parts we divided the test set""",
    )

    parser.add_argument(
        "--base-idx",
        default=1,
        type=int,
        help="""index of the first job, typically 0 or 1""",
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
    accum_speech_quality_metrics(**kwargs)


if __name__ == "__main__":
    main()
