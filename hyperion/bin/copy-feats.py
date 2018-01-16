#!/usr/bin/env python
"""
Copy features/vectors and change format
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import time
import numpy as np
from six.moves import xrange

from hyperion.io import CopyFeats as CF


if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Copy features and change format')

    parser.add_argument('--input', dest='input_spec', nargs='+', required=True)
    parser.add_argument('--output', dest='output_spec', required=True)
    #parser.add_argument('--write-num-frames', dest='write_num_frames', default=None)
    CF.add_argparse_args(parser)
    args=parser.parse_args()

    CF(**vars(args))
    
