"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import copy

import numpy as np
import h5py

from .list_utils import *


class TrialNdx(object):
    """Contains the trial index to run speaker recognition trials.
        Bosaris compatible Ndx.
    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      trial_mask: Boolean matrix with the trials to execute to True (num_models x num_segments).
    """

    def __init__(self, model_set=None, seg_set=None, trial_mask=None):
        self.model_set = model_set
        self.seg_set = seg_set
        self.trial_mask = trial_mask
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
        self.trial_mask = self.trial_mask[np.ix_(m_idx, s_idx)]

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
            f.create_dataset("trial_mask", data=self.trial_mask.astype("uint8"))

            # model_set = self.model_set.astype('S')
            # f.create_dataset('ID/row_ids', self.model_set.shape, dtype=model_set.dtype)
            # f['ID/row_ids'] = model_set
            # seg_set = self.seg_set.astype('S')
            # f.create_dataset('ID/column_ids', self.seg_set.shape, dtype=seg_set.dtype)
            # f['ID/column_ids'] = seg_set
            # f.create_dataset('trial_mask', self.trial_mask.shape, dtype='uint8')
            # f['trial_mask'] = self.trial_mask.astype('uint8')

    def save_txt(self, file_path):
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        idx = (self.trial_mask.T == True).nonzero()
        with open(file_path, "w") as f:
            for item in zip(idx[0], idx[1]):
                f.write("%s %s\n" % (self.model_set[item[1]], self.seg_set[item[0]]))

    @classmethod
    def load(cls, file_path):
        """Loads object from txt/h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialNdx object.
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
          TrialNdx object.
        """
        with h5py.File(file_path, "r") as f:
            model_set = [t.decode("utf-8") for t in f["ID/row_ids"]]
            seg_set = [t.decode("utf-8") for t in f["ID/column_ids"]]
            trial_mask = np.asarray(f["trial_mask"], dtype="bool")
        return cls(model_set, seg_set, trial_mask)

    @classmethod
    def load_txt(cls, file_path):
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          TrialNdx object.
        """
        with open(file_path, "r") as f:
            fields = [line.split() for line in f]
        models = [i[0] for i in fields]
        segments = [i[1] for i in fields]
        model_set, _, model_idx = np.unique(
            models, return_index=True, return_inverse=True
        )
        seg_set, _, seg_idx = np.unique(
            segments, return_index=True, return_inverse=True
        )
        trial_mask = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        for item in zip(model_idx, seg_idx):
            trial_mask[item[0], item[1]] = True
        return cls(model_set, seg_set, trial_mask)

    @classmethod
    def merge(cls, ndx_list):
        """Merges several index objects.

        Args:
          key_list: List of TrialNdx objects.

        Returns:
          Merged TrialNdx object.
        """
        num_ndx = len(ndx_list)
        model_set = ndx_list[0].model_set
        seg_set = ndx_list[0].seg_set
        trial_mask = ndx_list[0].trial_mask
        for i in range(1, num_ndx):
            ndx_i = ndx_list[i]
            new_model_set = np.union1d(model_set, ndx_i.model_set)
            new_seg_set = np.union1d(seg_set, ndx_i.seg_set)
            trial_mask_1 = np.zeros(
                (len(new_model_set), len(new_seg_set)), dtype="bool"
            )
            _, mi_a, mi_b = intersect(
                new_model_set, model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, seg_set, assume_unique=True, return_index=True
            )
            trial_mask_1[np.ix_(mi_a, si_a)] = trial_mask[np.ix_(mi_b, si_b)]

            trial_mask_2 = np.zeros(
                (len(new_model_set), len(new_seg_set)), dtype="bool"
            )
            _, mi_a, mi_b = intersect(
                new_model_set, ndx_i.model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, ndx_i.seg_set, assume_unique=True, return_index=True
            )
            trial_mask_2[np.ix_(mi_a, si_a)] = ndx_i.trial_mask[np.ix_(mi_b, si_b)]

            model_set = new_model_set
            seg_set = new_seg_set
            trial_mask = np.logical_or(trial_mask_1, trial_mask_2)

        return cls(model_set, seg_set, trial_mask)

    @staticmethod
    def parse_eval_set(ndx, enroll, test=None, eval_set="enroll-test"):
        """Prepares the data structures required for evaluation.

        Args:
          ndx: TrialNdx object cotaining the trials for the main evaluation.
          enroll: Utt2Info where key are file_ids and second column are model names
          test: Utt2Info of where key are test segments names.
                Needed in the cases enroll-coh and coh-coh.
          eval_test: Type of of evaluation
            enroll-test: main evaluation of enrollment vs test segments.
            enroll-coh: enrollment vs cohort segments.
            coh-test: cohort vs test segments.
            coh-coh: cohort vs cohort segments.

        Return:
          ndx: TrialNdx object
          enroll: SCPList
        """
        if eval_set == "enroll-test":
            enroll = enroll.filter_info(ndx.model_set)
        if eval_set == "enroll-coh":
            ndx = TrialNdx(ndx.model_set, test.file_path)
            enroll = enroll.filter_info(ndx.model_set)
        if eval_set == "coh-test":
            ndx = TrialNdx(enroll.key, ndx.seg_set)
        if eval_set == "coh-coh":
            ndx = TrialNdx(enroll.key, test.file_path)
        return ndx, enroll

    def filter(self, model_set, seg_set, keep=True):
        """Removes elements from TrialNdx object.

        Args:
          model_set: List of models to keep or remove.
          seg_set: List of test segments to keep or remove.
          keep: If True, we keep the elements in model_set/seg_set,
                if False, we remove the elements in model_set/seg_set.

        Returns:
          Filtered TrialNdx object.
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
        trial_mask = self.trial_mask[np.ix_(mod_idx, seg_idx)]
        return TrialNdx(model_set, seg_set, trial_mask)

    def split(self, model_idx, num_model_parts, seg_idx, num_seg_parts):
        """Splits the TrialNdx into num_model_parts x num_seg_parts and returns part
           (model_idx, seg_idx).

        Args:
          model_idx: Model index of the part to return from 1 to num_model_parts.
          num_model_parts: Number of parts to split the model list.
          seg_idx: Segment index of the part to return from 1 to num_model_parts.
          num_seg_parts: Number of parts to split the test segment list.

        Returns:
          Subpart of the TrialNdx
        """
        model_set, model_idx1 = split_list(self.model_set, model_idx, num_model_parts)
        seg_set, seg_idx1 = split_list(self.seg_set, seg_idx, num_seg_parts)
        trial_mask = self.trial_mask[np.ix_(model_idx1, seg_idx1)]
        return TrialNdx(model_set, seg_set, trial_mask)

    def validate(self):
        """Validates the attributes of the TrialKey object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        assert len(np.unique(self.model_set)) == len(self.model_set)
        assert len(np.unique(self.seg_set)) == len(self.seg_set)
        if self.trial_mask is None:
            self.trial_mask = np.ones(
                (len(self.model_set), len(self.seg_set)), dtype="bool"
            )
        else:
            assert self.trial_mask.shape == (len(self.model_set), len(self.seg_set))

    def apply_segmentation_to_test(self, segment_list):
        """Splits test segment into multiple sub-segments
        Useful to create ndx for spk diarization or tracking.

        Args:
          segment_list: ExtSegmentList object with mapping of
                        file_id to ext_segment_id
        Returns:
          New TrialNdx object with segment_ids in test instead of file_id.
        """
        new_segset = []
        new_mask = []
        for i in range(self.num_tests):
            file_id = self.seg_set[i]
            segment_ids = segment_list.ext_segment_ids_from_file(file_id)
            new_segset.append(segment_ids)
            new_mask.append(
                np.repeat(self.trial_mask[:, i, None], len(segment_ids), axis=1)
            )

        new_segset = np.concatenate(tuple(new_segset))
        new_mask = np.concatenate(tuple(new_mask), axis=-1)
        return TrialNdx(self.model_set, new_segset, new_mask)

    def __eq__(self, other):
        """Equal operator"""
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(self.trial_mask == other.trial_mask)
        return eq

    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(oher):
            return 0
        return 1

    def test(ndx_file="core-core_det5_ndx.h5"):

        ndx1 = TrialNdx.load(ndx_file)
        ndx1.sort()
        ndx2 = ndx1.copy()

        ndx2.model_set[0] = "m1"
        ndx2.trial_mask[:] = 0
        assert np.any(ndx1.model_set != ndx2.model_set)
        assert np.any(ndx1.trial_mask != ndx2.trial_mask)

        ndx2 = TrialNdx(ndx1.model_set[:10], ndx1.seg_set, ndx1.trial_mask[:10, :])
        ndx3 = TrialNdx(ndx1.model_set[5:], ndx1.seg_set, ndx1.trial_mask[5:, :])
        ndx4 = TrialNdx.merge([ndx2, ndx3])
        assert ndx1 == ndx4

        ndx2 = TrialNdx(ndx1.model_set, ndx1.seg_set[:10], ndx1.trial_mask[:, :10])
        ndx3 = TrialNdx(ndx1.model_set, ndx1.seg_set[5:], ndx1.trial_mask[:, 5:])
        ndx4 = TrialNdx.merge([ndx2, ndx3])
        assert ndx1 == ndx4

        ndx2 = TrialNdx(ndx1.model_set[:5], ndx1.seg_set[:10], ndx1.trial_mask[:5, :10])
        ndx3 = ndx1.filter(ndx2.model_set, ndx2.seg_set, keep=True)
        assert ndx2 == ndx3

        num_parts = 3
        ndx_list = []
        for i in range(num_parts):
            for j in range(num_parts):
                ndx_ij = ndx1.split(i + 1, num_parts, j + 1, num_parts)
                ndx_list.append(ndx_ij)
        ndx2 = TrialNdx.merge(ndx_list)
        assert ndx1 == ndx2

        file_h5 = "test.h5"
        ndx1.save(file_h5)
        ndx2 = TrialNdx.load(file_h5)
        assert ndx1 == ndx2

        file_txt = "test.txt"
        ndx3.trial_mask[0, :] = True
        ndx3.trial_mask[:, 0] = True
        ndx3.save(file_txt)
        ndx2 = TrialNdx.load(file_txt)
        assert ndx3 == ndx2
