#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import logging
import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import SparseTrialKey


def make_trials(in_key_file, out_key_file, ntar, nnon, seed):

    rng = np.random.RandomState(seed=seed)

    logging.info("Load key: %s" % in_key_file)
    key = SparseTrialKey.load_txt(in_key_file)

    nz_idx = key.tar.nonzero()
    nnz = len(nz_idx[0])
    p = rng.permutation(nnz)[ntar:]
    nz_idx = (nz_idx[0][p], nz_idx[1][p])
    key.tar[nz_idx] = False

    nz_idx = key.non.nonzero()
    nnz = len(nz_idx[0])
    p = rng.permutation(nnz)[nnon:]
    nz_idx = (nz_idx[0][p], nz_idx[1][p])
    key.non[nz_idx] = False

    logging.info("Saving key: %s" % out_key_file)
    key.save_txt(out_key_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Makes a subset of a trial key")

    parser.add_argument("--in-key-file", required=True)
    parser.add_argument("--out-key-file", required=True)
    parser.add_argument("--ntar", required=True, type=int)
    parser.add_argument("--nnon", required=True, type=int)
    parser.add_argument("--seed", default=112358, type=int)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    make_trials(**namespace_to_dict(args))
