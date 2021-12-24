"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import numpy as np
import h5py

from ..hyp_defs import float_save
from ..utils.list_utils import list2ndarray, ismember


class HypDataWriter(object):
    """
    Class to write data to hdf5 files (deprecated).
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(file_path, "w")

    def write(self, keys, field, x):
        # datasets = keys.astype(np.object)+field
        if isinstance(keys, str):
            keys = [keys]
            x = [x]

        datasets = [key + field for key in keys]
        for i, ds in enumerate(datasets):
            self.f.create_dataset(ds, data=x[i].astype(float_save()))
