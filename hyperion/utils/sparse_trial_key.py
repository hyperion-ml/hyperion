"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import copy

import numpy as np
import scipy.sparse as sparse

from .list_utils import *
from .trial_ndx import TrialNdx
from .trial_key import TrialKey


class SparseTrialKey(TrialKey):

    """Contains the trial key for speaker recognition trials.
        Bosaris compatible Key.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      tar: Boolean matrix with target trials to True (num_models x num_segments).
      non: Boolean matrix with non-target trials to True (num_models x num_segments).
      model_cond: Conditions related to the model.
      seg_cond: Conditions related to the test segment.
      trial_cond: Conditions related to the combination of model and test segment.
      model_cond_name: String list with the names of the model conditions.
      seg_cond_name: String list with the names of the segment conditions.
      trial_cond_name: String list with the names of the trial conditions.
    """

    def __init__(
        self,
        model_set=None,
        seg_set=None,
        tar=None,
        non=None,
        model_cond=None,
        seg_cond=None,
        trial_cond=None,
        model_cond_name=None,
        seg_cond_name=None,
        trial_cond_name=None,
    ):

        super().__init__(
            model_set,
            seg_set,
            tar,
            non,
            model_cond,
            seg_cond,
            trial_cond,
            model_cond_name,
            seg_cond_name,
            trial_cond_name,
        )

    def save_h5(self, file_path):
        raise NotImplementedError()

    def save_txt(self, file_path):
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        with open(file_path, "w") as f:
            self.tar.eliminate_zeros()
            self.non.eliminate_zeros()
            tar = self.tar.tocoo()
            for r, c in zip(tar.row, tar.col):
                f.write("%s %s target\n" % (self.model_set[r], self.seg_set[c]))

            non = self.non.tocoo()
            for r, c in zip(non.row, non.col):
                f.write("%s %s nontarget\n" % (self.model_set[r], self.seg_set[c]))

    @classmethod
    def load_h5(cls, file_path):
        raise NotImplementedError()

    @classmethod
    def load_txt(cls, file_path):
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          TrialKey object.
        """
        with open(file_path, "r") as f:
            fields = [line.split() for line in f]
        models = [i[0] for i in fields]
        segments = [i[1] for i in fields]
        is_tar = [i[2] == "target" for i in fields]
        model_set, _, model_idx = np.unique(
            models, return_index=True, return_inverse=True
        )
        seg_set, _, seg_idx = np.unique(
            segments, return_index=True, return_inverse=True
        )
        tar = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        non = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        for item in zip(model_idx, seg_idx, is_tar):
            if item[2]:
                tar[item[0], item[1]] = True
            else:
                non[item[0], item[1]] = True
        return cls(model_set, seg_set, tar.tocsr(), non.tocsr())

    @classmethod
    def merge(cls, key_list):
        raise NotImplementedError()

    def to_ndx(self):
        """Converts TrialKey object into TrialNdx object.

        Returns:
          TrialNdx object.
        """
        mask = np.logical_or(self.tar.toarray(), self.non.toarray())
        return TrialNdx(self.model_set, self.seg_set, mask)

    def validate(self):
        """Validates the attributes of the TrialKey object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        shape = (len(self.model_set), len(self.seg_set))
        assert len(np.unique(self.model_set)) == shape[0]
        assert len(np.unique(self.seg_set)) == shape[1]

        if (self.tar is None) or (self.non is None):
            self.tar = sparse.csr_matrix(shape, dtype="bool")
            self.non = sparse.csr_matrix(shape, dtype="bool")
        else:
            assert self.tar.shape == shape
            assert self.non.shape == shape

        if self.model_cond is not None:
            assert self.model_cond.shape[1] == shape[0]
        if self.seg_cond is not None:
            assert self.seg_cond.shape[1] == shape[1]
        if self.trial_cond is not None:
            assert self.trial_cond.shape[1:] == shape

        if self.model_cond_name is not None:
            self.model_cond_name = list2ndarray(self.model_cond_name)
        if self.seg_cond_name is not None:
            self.seg_cond_name = list2ndarray(self.seg_cond_name)
        if self.trial_cond_name is not None:
            self.trial_cond_name = list2ndarray(self.trial_cond_name)

    @classmethod
    def from_trial_key(cls, key):
        tar = sparse.csr_matrix(key.tar)
        non = sparse.csr_matrix(key.non)
        tar.eliminate_zeros()
        non.eliminate_zeros()
        return cls(
            key.model_set,
            key.seg_set,
            tar,
            non,
            key.model_cond,
            key.seg_cond,
            key.trial_cond,
            key.model_cond_name,
            key.seg_cond_name,
            key.trial_cond_name,
        )

    def __eq__(self, other):
        """Equal operator"""

        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(self.tar.data == other.tar.data)
        eq = eq and np.all(self.non.data == other.non.data)
        eq = eq and np.all(self.tar.indices == other.tar.indices)
        eq = eq and np.all(self.non.indices == other.non.indices)

        eq = eq and ((self.model_cond is None) == (other.model_cond is None))
        eq = eq and ((self.seg_cond is None) == (other.seg_cond is None))
        eq = eq and ((self.trial_cond is None) == (other.trial_cond is None))

        if self.model_cond is not None:
            eq = eq and np.all(self.model_cond == other.model_cond)
        if self.seg_cond is not None:
            eq = eq and np.all(self.seg_cond == other.seg_cond)
        if self.trial_cond is not None:
            eq = eq and np.all(self.triall_cond == other.trial_cond)

        eq = eq and ((self.model_cond_name is None) == (other.model_cond_name is None))
        eq = eq and ((self.seg_cond_name is None) == (other.seg_cond_name is None))
        eq = eq and ((self.trial_cond_name is None) == (other.trial_cond_name is None))

        if self.model_cond_name is not None:
            eq = eq and np.all(self.model_cond_name == other.model_cond_name)
        if self.seg_cond_name is not None:
            eq = eq and np.all(self.seg_cond_name == other.seg_cond_name)
        if self.trial_cond_name is not None:
            eq = eq and np.all(self.triall_cond_name == other.trial_cond_name)

        return eq
