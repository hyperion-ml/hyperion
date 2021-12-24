"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import numpy as np
import h5py

from ..hyp_defs import float_cpu
from ..utils.list_utils import list2ndarray, ismember


class HypDataReader(object):
    """
    Class to read data from hdf5 files (deprecated).
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(file_path, "r")

    def get_datasets(self):
        keys = []
        for ds in self.f:
            keys.append(ds)
        return keys

    def read(self, keys, field="", return_tensor=False):
        if isinstance(keys, list):
            datasets = [key + field for key in keys]
        else:
            datasets = keys.astype(np.object) + field

        if return_tensor:
            # we assume that all datasets have a common shape
            shape_0 = self.f[datasets[0]].shape
            shape = tuple([len(keys)] + list(shape_0))
            X = np.zeros(shape, dtype=float_cpu())
        else:
            X = []

        for i in range(len(keys)):
            assert datasets[i] in self.f, "Dataset %s not found" % datasets[i]
            X_i = self.f[datasets[i]]
            if return_tensor:
                X[i] = X_i
            else:
                X.append(np.asarray(X_i, dtype=float_cpu()))

        return X

    def get_num_rows(self, keys, field=""):
        if isinstance(keys, list):
            datasets = [key + field for key in keys]
        else:
            datasets = keys.astype(np.object) + field

        num_ds = len(datasets)
        num_rows = np.zeros((num_ds,), dtype=int)

        for i, ds in enumerate(datasets):
            assert ds in self.f, "Dataset %s not found" % ds
            num_rows[i] = self.f[ds].shape[0]

        return num_rows

    def read_slice(self, key, index, num_samples, field=""):
        dataset = key + field
        assert dataset in self.f, "Dataset %s not found" % dataset
        X = self.f[dataset][index : index + num_samples]
        return X

    def read_random_slice(self, key, num_samples, rng, field=""):
        dataset = key + field
        assert dataset in self.f, "Dataset %s not found" % dataset
        num_rows = self.f[dataset].shape[0]
        # print('hola',num_rows,num_samples,num_rows-num_samples)
        # index = rng.random_integers(low=0, high=num_rows-num_samples, size=1)[0]
        index = rng.randint(low=0, high=num_rows - num_samples + 1)
        X = self.f[dataset][index : index + num_samples]
        return X, index

    def read_random_samples(self, key, num_samples, rng, field="", replace=True):
        dataset = key + field
        assert dataset in self.f, "Dataset %s not found" % dataset
        num_rows = self.f[dataset].shape[0]
        index = np.sort(
            rng.choice(np.arange(num_rows), size=num_samples, replace=replace)
        )
        min_index = index[0]
        max_index = index[-1] + 1
        index -= min_index
        X = self.f[dataset][min_index:max_index]
        X = X[index]
        return X, index
