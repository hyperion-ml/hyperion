#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path

from hyperion.hyp_defs import config_logger
from hyperion.utils import Dataset
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            """Split speakers in dataset into test speaker to create ASV trials and 
        cohort speakers for S-Norm"""
        )
    )

    parser.add_argument("--data-dir", required=True, help="Path to dataset")
    parser.add_argument(
        "--num-1k-tar-trials", type=int, default=30, help="thousands of target trials"
    )
    parser.add_argument(
        "--num-trial-speakers",
        type=int,
        default=1000,
        help="number of speakers to create trials",
    )
    parser.add_argument(
        "--intra-gender",
        default=True,
        action=ActionYesNo,
        help="Whether we create intra gender trials or not",
    )
    parser.add_argument("--seed", type=int, default=1123, help="random seed")
    parser.add_argument(
        "--trials-dir", default=None, help="Path to output trials dataset"
    )
    parser.add_argument(
        "--cohort-dir", default=None, help="Path to output cohort dataset"
    )

    args = parser.parse_args()
    config_logger(1)
    data_dir = args.data_dir
    cohort_dir = args.cohort_dir
    cohort_dir = f"{data_dir}_cohort" if cohort_dir is None else cohort_dir
    trials_dir = args.trials_dir
    trials_dir = f"{data_dir}_trials" if trials_dir is None else trials_dir

    del args.data_dir
    del args.cohort_dir
    del args.trials_dir
    args = namespace_to_dict(args)

    dataset = Dataset.load(data_dir)
    trials_dataset, cohort_dataset = dataset.split_into_trials_and_cohort(**args)
    trials_dataset.save(trials_dir)
    cohort_dataset.save(cohort_dir)
