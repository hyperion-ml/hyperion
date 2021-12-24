"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import numpy as np

from .hyp_data_reader import HypDataReader as HR
from .hyp_data_writer import HypDataWriter as HW


class H5Merger(object):
    """Merges several hdf5 files into one."""

    def __init__(self, input_files, output_file, chunk_size=None):
        self.input_files = input_files
        self.output_file = output_file
        self.chunk_size = chunk_size

    def merge(self):
        hw = HW(self.output_file)
        for h5_file in self.input_files:
            self._merge_file(hw, h5_file)

    def _merge_file(self, hw, input_file):
        hr = HR(input_file)
        datasets = hr.get_datasets()
        if self.chunk_size is None:
            chunk = len(datasets)
        else:
            chunk = self.chunk_size

        for first in range(0, len(datasets), chunk):
            last = min(first + chunk, len(datasets))
            keys = datasets[first:last]
            x = hr.read(keys)
            hw.write(keys, "", x)
