#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

from pathlib import Path

import numpy as np
import yaml

from hyperion.hyp_defs import float_cpu, config_logger


def filter_attacks(input_file, output_file, field, keep, remove):

    logging.info("reading {}".format(input_file))
    with open(input_file, "r") as f:
        attack_info = yaml.load(f, Loader=yaml.FullLoader)

    logging.info("selecting elements to remove")
    rem_list = []
    for k, v in attack_info.items():
        if not (v[field] in keep) or v[field] in remove:
            rem_list.append(k)

    logging.info("removing elements")
    [attack_info.pop(k) for k in rem_list]

    logging.info("saving {}".format(output_file))
    with open(output_file, "w") as f:
        yaml.dump(attack_info, f, sort_keys=True)


if __name__ == "__main__":

    parser = ArgumentParser(description="Filters attacks in yaml file")

    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--keep", nargs="+", default=[])
    parser.add_argument("--remove", nargs="+", default=[])
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    filter_attacks(**namespace_to_dict(args))
