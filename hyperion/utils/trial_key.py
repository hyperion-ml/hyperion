"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import copy

import numpy as np
import h5py

from .list_utils import *
from .trial_ndx import TrialNdx


class TrialKey(object):
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
        self.model_set = model_set
        self.seg_set = seg_set
        self.tar = tar
        self.non = non
        self.model_cond = model_cond
        self.seg_cond = seg_cond
        self.trial_cond = trial_cond
        self.model_cond_name = model_cond_name
        self.seg_cond_name = seg_cond_name
        self.trial_cond_name = trial_cond_name
        if (model_set is not None) and (seg_set is not None):
            self.validate()

    @property
    def num_models(self):
        return len(self.model_set)

    @property
    def num_tests(self):
        return len(self.seg_set)

    def copy(self):
        """Makes a copy of the object"""
        return copy.deepcopy(self)

    def sort(self):
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        ix = np.ix_(m_idx, s_idx)
        self.tar = self.tar[ix]
        self.non = self.non[ix]
        if self.model_cond is not None:
            self.model_cond = self.model_cond[m_idx]
        if self.seg_cond is not None:
            self.seg_cond = self.seg_cond[s_idx]
        if self.trial_cond is not None:
            self.trial_cond = self.trial_cond[:, ix]

    def save(self, file_path):
        """Saves object to txt/h5 file.

        Args:
          file_path: File to write the list.
        """

        file_base, file_ext = path.splitext(file_path)
        if file_ext == ".h5" or file_ext == ".hdf5":
            self.save_h5(file_path)
        else:
            self.save_txt(file_path)

    def save_h5(self, file_path):
        """Saves object to h5 file.

        Args:
          file_path: File to write the list.
        """

        with h5py.File(file_path, "w") as f:
            model_set = self.model_set.astype("S")
            seg_set = self.seg_set.astype("S")
            f.create_dataset("ID/row_ids", data=model_set)
            f.create_dataset("ID/column_ids", data=seg_set)
            trial_mask = self.tar.astype("int8") - self.non.astype("int8")
            f.create_dataset("trial_mask", data=trial_mask)
            if self.model_cond is not None:
                f.create_dataset("model_cond", data=self.model_cond.astype("uint8"))
            if self.seg_cond is not None:
                f.create_dataset("seg_cond", data=self.seg_cond.astype("uint8"))
            if self.trial_cond is not None:
                f.create_dataset("trial_cond", data=self.trial_cond.astype("uint8"))
            if self.model_cond_name is not None:
                model_cond_name = self.model_cond_name.astype("S")
                f.create_dataset("model_cond_name", data=model_cond_name)
            if self.seg_cond_name is not None:
                seg_cond_name = self.seg_cond_name.astype("S")
                f.create_dataset("seg_cond_name", data=seg_cond_name)
            if self.trial_cond_name is not None:
                trial_cond_name = self.trial_cond_name.astype("S")
                f.create_dataset("trial_cond_name", data=trial_cond_name)

    def save_txt(self, file_path):
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        with open(file_path, "w") as f:
            idx = (self.tar.T == True).nonzero()
            for item in zip(idx[0], idx[1]):
                f.write(
                    "%s %s target\n" % (self.model_set[item[1]], self.seg_set[item[0]])
                )
            idx = (self.non.T == True).nonzero()
            for item in zip(idx[0], idx[1]):
                f.write(
                    "%s %s nontarget\n"
                    % (self.model_set[item[1]], self.seg_set[item[0]])
                )

    @classmethod
    def load(cls, file_path):
        """Loads object from txt/h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialKey object.
        """
        file_base, file_ext = path.splitext(file_path)
        if file_ext == ".h5" or file_ext == ".hdf5":
            return cls.load_h5(file_path)
        else:
            return cls.load_txt(file_path)

    @classmethod
    def load_h5(cls, file_path):
        """Loads object from h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialKey object.
        """
        with h5py.File(file_path, "r") as f:
            model_set = [t.decode("utf-8") for t in f["ID/row_ids"]]
            seg_set = [t.decode("utf-8") for t in f["ID/column_ids"]]

            trial_mask = np.asarray(f["trial_mask"], dtype="int8")
            tar = (trial_mask > 0).astype("bool")
            non = (trial_mask < 0).astype("bool")

            model_cond = None
            seg_cond = None
            trial_cond = None
            model_cond_name = None
            seg_cond_name = None
            trial_cond_name = None
            if "model_cond" in f:
                model_cond = np.asarray(f["model_cond"], dtype="bool")
            if "seg_cond" in f:
                seg_cond = np.asarray(f["seg_cond"], dtype="bool")
            if "trial_cond" in f:
                trial_cond = np.asarray(f["trial_cond"], dtype="bool")
            if "model_cond_name" in f:
                model_cond_name = np.asarray(f["model_cond_name"], dtype="U")
            if "seg_cond_name" in f:
                seg_cond_name = np.asarray(f["seg_cond_name"], dtype="U")
            if "trial_cond_name" in f:
                trial_cond_name = np.asarray(f["trial_cond_name"], dtype="U")

        return cls(
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
        tar = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        non = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        for item in zip(model_idx, seg_idx, is_tar):
            if item[2]:
                tar[item[0], item[1]] = True
            else:
                non[item[0], item[1]] = True
        return cls(model_set, seg_set, tar, non)

    @classmethod
    def merge(cls, key_list):
        """Merges several key objects.

        Args:
          key_list: List of TrialKey objects.

        Returns:
          Merged TrialKey object.
        """
        num_key = len(key_list)
        model_set = key_list[0].model_set
        seg_set = key_list[0].seg_set
        tar = key_list[0].tar
        non = key_list[0].non
        model_cond = key_list[0].model_cond
        seg_cond = key_list[0].seg_cond
        trial_cond = key_list[0].trial_cond

        if model_cond is not None:
            num_model_cond = model_cond.shape[0]
        if seg_cond is not None:
            num_seg_cond = seg_cond.shape[0]
        if trial_cond is not None:
            num_trial_cond = trial_cond.shape[0]

        for i in range(1, num_key):
            key_i = key_list[i]
            new_model_set = np.union1d(model_set, key_i.model_set)
            new_seg_set = np.union1d(seg_set, key_i.seg_set)
            shape = (len(new_model_set), len(new_seg_set))

            _, mi_a, mi_b = intersect(
                new_model_set, model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, seg_set, assume_unique=True, return_index=True
            )
            ix_a = np.ix_(mi_a, si_a)
            ix_b = np.ix_(mi_b, si_b)
            tar_1 = np.zeros(shape, dtype="bool")
            tar_1[ix_a] = tar[ix_b]
            non_1 = np.zeros(shape, dtype="bool")
            non_1[ix_a] = non[ix_b]
            if model_cond is not None:
                model_cond_1 = np.zeros((num_model_cond, shape[0]), dtype="bool")
                model_cond_1[:, mi_a] = model_cond[:, mi_b]
            if seg_cond is not None:
                seg_cond_1 = np.zeros((num_seg_cond, shape[0]), dtype="bool")
                seg_cond_1[:, mi_a] = seg_cond[:, mi_b]
            if trial_cond is not None:
                trial_cond_1 = np.zeros((num_trial_cond, shape), dtype="bool")
                trial_cond_1[:, ix_a] = trial_cond[:, ix_b]

            _, mi_a, mi_b = intersect(
                new_model_set, key_i.model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, key_i.seg_set, assume_unique=True, return_index=True
            )
            ix_a = np.ix_(mi_a, si_a)
            ix_b = np.ix_(mi_b, si_b)
            tar_2 = np.zeros(shape, dtype="bool")
            tar_2[ix_a] = key_i.tar[ix_b]
            non_2 = np.zeros(shape, dtype="bool")
            non_2[ix_a] = key_i.non[ix_b]
            if model_cond is not None:
                model_cond_2 = np.zeros((num_model_cond, shape[0]), dtype="bool")
                model_cond_2[:, mi_a] = key_i.model_cond[:, mi_b]
            if seg_cond is not None:
                seg_cond_2 = np.zeros((num_seg_cond, shape[0]), dtype="bool")
                seg_cond_2[:, mi_a] = key_i.seg_cond[:, mi_b]
            if trial_cond is not None:
                trial_cond_2 = np.zeros((num_trial_cond, shape), dtype="bool")
                trial_cond_2[:, ix_a] = key_i.trial_cond[:, ix_b]

            model_set = new_model_set
            seg_set = new_seg_set
            tar = np.logical_or(tar_1, tar_2)
            non = np.logical_or(non_1, non_2)
            if model_cond is not None:
                model_cond = np.logical_or(model_cond_1, model_cond_2)
            if seg_cond is not None:
                seg_cond = np.logical_or(seg_cond_1, seg_cond_2)
            if trial_cond is not None:
                trial_cond = np.logical_or(trial_cond_1, seg_cond_2)

        return cls(
            model_set,
            seg_set,
            tar,
            non,
            model_cond,
            seg_cond,
            trial_cond,
            key_list[0].model_cond_name,
            key_list[0].seg_cond_name,
            key_list[0].trial_cond_name,
        )

    def filter(self, model_set, seg_set, keep=True):
        """Removes elements from TrialKey object.

        Args:
          model_set: List of models to keep or remove.
          seg_set: List of test segments to keep or remove.
          keep: If True, we keep the elements in model_set/seg_set,
                if False, we remove the elements in model_set/seg_set.

        Returns:
          Filtered TrialKey object.
        """

        if not (keep):
            model_set = np.setdiff1d(self.model_set, model_set)
            seg_set = np.setdiff1d(self.seg_set, seg_set)

        f, mod_idx = ismember(model_set, self.model_set)
        assert np.all(f)
        f, seg_idx = ismember(seg_set, self.seg_set)
        assert np.all(f)

        model_set = self.model_set[mod_idx]
        set_set = self.seg_set[seg_idx]
        ix = np.ix_(mod_idx, seg_idx)
        tar = self.tar[ix]
        non = self.non[ix]

        model_cond = None
        seg_cond = None
        trial_cond = None
        if self.model_cond is not None:
            model_cond = self.model_cond[:, mod_idx]
        if self.seg_cond is not None:
            seg_cond = self.seg_cond[:, seg_idx]
        if self.trial_cond is not None:
            trial_cond = self.trial_cond[:, ix]

        return TrialKey(
            model_set,
            seg_set,
            tar,
            non,
            model_cond,
            seg_cond,
            trial_cond,
            self.model_cond_name,
            self.seg_cond_name,
            self.trial_cond_name,
        )

    def split(self, model_idx, num_model_parts, seg_idx, num_seg_parts):
        """Splits the TrialKey into num_model_parts x num_seg_parts and returns part
           (model_idx, seg_idx).

        Args:
          model_idx: Model index of the part to return from 1 to num_model_parts.
          num_model_parts: Number of parts to split the model list.
          seg_idx: Segment index of the part to return from 1 to num_model_parts.
          num_seg_parts: Number of parts to split the test segment list.

        Returns:
          Subpart of the TrialKey
        """

        model_set, model_idx1 = split_list(self.model_set, model_idx, num_model_parts)
        seg_set, seg_idx1 = split_list(self.seg_set, seg_idx, num_seg_parts)
        ix = np.ix_(model_idx1, seg_idx1)
        tar = self.tar[ix]
        non = self.non[ix]

        model_cond = None
        seg_cond = None
        trial_cond = None
        if self.model_cond is not None:
            model_cond = self.model_cond[:, model_idx1]
        if self.seg_cond is not None:
            seg_cond = self.seg_cond[:, seg_idx1]
        if self.trial_cond is not None:
            trial_cond = self.trial_cond[:, ix]

        return TrialKey(
            model_set,
            seg_set,
            tar,
            non,
            model_cond,
            seg_cond,
            trial_cond,
            self.model_cond_name,
            self.seg_cond_name,
            self.trial_cond_name,
        )

    def to_ndx(self):
        """Converts TrialKey object into TrialNdx object.

        Returns:
          TrialNdx object.
        """
        mask = np.logical_or(self.tar, self.non)
        return TrialNdx(self.model_set, self.seg_set, mask)

    def validate(self):
        """Validates the attributes of the TrialKey object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        shape = (len(self.model_set), len(self.seg_set))
        assert len(np.unique(self.model_set)) == shape[0]
        assert len(np.unique(self.seg_set)) == shape[1]

        if (self.tar is None) or (self.non is None):
            self.tar = np.zeros(shape, dtype="bool")
            self.non = np.zeros(shape, dtype="bool")
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

    def __eq__(self, other):
        """Equal operator"""

        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(self.tar == other.tar)
        eq = eq and np.all(self.non == other.non)

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

    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1

    def test(key_file="core-core_det5_key.h5"):

        key1 = TrialKey.load(key_file)
        key1.sort()
        key2 = key1.copy()

        key2.model_set[0] = "m1"
        key2.tar[:] = 0
        assert np.any(key1.model_set != key2.model_set)
        assert np.any(key1.tar != key2.tar)

        key2 = TrialKey(
            key1.model_set[:10], key1.seg_set, key1.tar[:10, :], key1.non[:10, :]
        )
        key3 = TrialKey(
            key1.model_set[5:], key1.seg_set, key1.tar[5:, :], key1.non[5:, :]
        )
        key4 = TrialKey.merge([key2, key3])
        assert key1 == key4

        key2 = TrialKey(
            key1.model_set, key1.seg_set[:10], key1.tar[:, :10], key1.non[:, :10]
        )
        key3 = TrialKey(
            key1.model_set, key1.seg_set[5:], key1.tar[:, 5:], key1.non[:, 5:]
        )
        key4 = TrialKey.merge([key2, key3])
        assert key1 == key4

        key2 = TrialKey(
            key1.model_set[:5], key1.seg_set[:10], key1.tar[:5, :10], key1.non[:5, :10]
        )
        key3 = key1.filter(key2.model_set, key2.seg_set, keep=True)
        assert key2 == key3

        num_parts = 3
        key_list = []
        for i in range(num_parts):
            for j in range(num_parts):
                key_ij = key1.split(i + 1, num_parts, j + 1, num_parts)
                key_list.append(key_ij)
        key2 = TrialKey.merge(key_list)
        assert key1 == key2

        ndx1 = key1.to_ndx()
        ndx1.validate()

        file_h5 = "test.h5"
        key1.save(file_h5)
        key3 = TrialKey.load(file_h5)
        assert key1 == key2

        file_txt = "test.txt"
        key3.tar[0, :] = True
        key3.non[:, 0] = True
        key3.save(file_txt)
        key2 = TrialKey.load(file_txt)
        assert key3 == key2
