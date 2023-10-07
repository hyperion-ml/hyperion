#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path

from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import TrialScores


def merge_scores(input_files, output_file, num_enroll_parts, num_test_parts, base_idx):
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    ext = output_file.suffix

    if input_files is None:
        if ext in [".h5", ".csv", ".tsv"]:
            input_file_base = output_file
        else:
            input_file_base = output_file.parent / (output_file.name + ".txt")
            ext = ""

        logging.info("merging %s* -> %s", input_file_base.with_suffix(""), output_file)
        input_files = []
        for i in range(num_enroll_parts):
            idx_i = base_idx + i
            for j in range(num_test_parts):
                idx_j = base_idx + j
                input_file_i = input_file_base.with_suffix(f".{idx_i}.{idx_j}{ext}")
                input_files.append(input_file_i)
    else:
        logging.info("merging %s -> %s", " + ".join(input_files), output_file)

    if ext == ".h5":
        # if files are h5 we need to load everything in RAM
        score_list = []
        for score_file in input_files:
            scores = TrialScores.load(score_file)
            score_list.append(scores)

        scores = TrialScores.merge(score_list)
        scores.save(output_file)
    else:
        has_header = ext in [".csv", ".tsv"]
        write_header = True
        with open(output_file, "w", encoding="utf-8") as f_out:
            for score_file in input_files:
                with open(score_file) as f_in:
                    for i, line in enumerate(f_in):
                        if i == 0 and has_header and not write_header:
                            continue
                        f_out.write(line)
                        write_header = False


def main():
    parser = ArgumentParser(description="Tool to manipulates the Hyperion data tables")
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
        "--num-enroll-parts",
        default=1,
        type=int,
        help="""number of parts we divided the enrollment set""",
    )
    parser.add_argument(
        "--num-test-parts",
        default=1,
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
    merge_scores(**kwargs)


if __name__ == "__main__":
    main()
