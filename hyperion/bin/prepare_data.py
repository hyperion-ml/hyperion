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

from hyperion.data_prep import DataPrep


def make_parser(data_prep_class):
    parser = ArgumentParser()
    data_prep_class.add_class_args(parser)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Prepares a dataset into relational database tables"""
    )
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for k, v in DataPrep.registry.items():
        parser_k = make_parser(v)
        subcommands.add_subcommand(k, parser_k)

    args = parser.parse_args()
    data_prep_class = DataPrep.registry[args.subcommand]
    args = namespace_to_dict(args)[args.subcommand]

    data_prep = data_prep_class(**args)
    data_prep.prepare()
