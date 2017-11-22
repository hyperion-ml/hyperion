"""
Classes to read data from hdf5 files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import sys
import numpy as np
import h5py

from ..hyp_defs import float_cpu
from ..utils.kaldi_io_funcs import is_token
from .data_reader import SequentialDataReader, RandomAccessDataReader



