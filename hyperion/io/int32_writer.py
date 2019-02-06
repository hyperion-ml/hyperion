"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import sys

from .data_writer import DataWriter

class Int32Writer(DataWriter):
    """Class to write data to int32 files.
    """
    def __init__(self, wspecifier):
        super(Int32Writer, self).__init__(wspecifier)
