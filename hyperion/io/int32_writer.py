"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys

from .data_writer import DataWriter


class Int32Writer(DataWriter):
    """Class to write data to int32 files."""

    def __init__(self, wspecifier):
        super(Int32Writer, self).__init__(wspecifier)
