"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import os.path as path
import logging
import copy

import numpy as np
import scipy.sparse as sparse

# import h5py

from ..hyp_defs import float_cpu
from .list_utils import *
from .trial_ndx import TrialNdx
from .trial_key import TrialKey
from .sparse_trial_key import SparseTrialKey
from .trial_scores import TrialScores


class SparseTrialScores(TrialScores):

    """Contains the scores for the speaker recognition trials.
        Bosaris compatible Scores.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      scores: Matrix with the scores (num_models x num_segments).
      score_mask: Boolean matrix with the trials with valid scores to True (num_models x num_segments).
    """

    def __init__(self, model_set=None, seg_set=None, scores=None, score_mask=None):
        super(SparseTrialScores, self).__init__(model_set, seg_set, scores, score_mask)

    def save_h5(self, file_path):
        raise NotImplementedError()

    def save_txt(self, file_path):
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        self.score_mask.eliminate_zeros()
        score_mask = self.score_mask.tocoo()
        with open(file_path, "w") as f:
            for r, c in zip(score_mask.row, score_mask.col):
                f.write(
                    "%s %s %f\n"
                    % (self.model_set[r], self.seg_set[c], self.scores[r, c])
                )

    @classmethod
    def load_h5(cls, file_path):
        raise NotImplementedError()

    @classmethod
    def load_txt(cls, file_path):
        """Loads object from h5 file

        Args:
          file_path: File to read the list.

        Returns:
          SparseTrialScores object.
        """
        with open(file_path, "r") as f:
            fields = [line.split() for line in f]

        models = [i[0] for i in fields]
        segments = [i[1] for i in fields]
        scores_v = np.array([i[2] for i in fields])

        model_set, _, model_idx = np.unique(
            models, return_index=True, return_inverse=True
        )
        seg_set, _, seg_idx = np.unique(
            segments, return_index=True, return_inverse=True
        )

        scores = sparse.lil_matrix((len(model_set), len(seg_set)), dtype=float_cpu())
        score_mask = sparse.lil_matrix(scores.shape, dtype="bool")
        for item in zip(model_idx, seg_idx, scores_v):
            score_mask[item[0], item[1]] = True
            scores[item[0], item[1]] = item[2]
        return cls(model_set, seg_set, scores.tocsr(), score_mask.tocsr())

    @classmethod
    def merge(cls, scr_list):
        raise NotImplementedError()

    def split(self, model_idx, num_model_parts, seg_idx, num_seg_parts):
        """Splits the TrialScores into num_model_parts x num_seg_parts and returns part
           (model_idx, seg_idx).

        Args:
          model_idx: Model index of the part to return from 1 to num_model_parts.
          num_model_parts: Number of parts to split the model list.
          seg_idx: Segment index of the part to return from 1 to num_model_parts.
          num_seg_parts: Number of parts to split the test segment list.

        Returns:
          Subpart of the TrialScores
        """

        model_set, model_idx1 = split_list(self.model_set, model_idx, num_model_parts)
        seg_set, seg_idx1 = split_list(self.seg_set, seg_idx, num_seg_parts)
        ix = np.ix_(model_idx1, seg_idx1)
        scores = self.scores[ix]
        score_mask = self.score_mask[ix]
        return SparseTrialScores(model_set, seg_set, scores, score_mask)

    def validate(self):
        """Validates the attributes of the TrialKey object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        assert len(np.unique(self.model_set)) == len(self.model_set)
        assert len(np.unique(self.seg_set)) == len(self.seg_set)
        if self.scores is None:
            self.scores = sparse.csr_matrix(
                (len(model_set), len(seg_set)), dtype=float_cpu()
            )
        else:
            assert self.scores.shape == (len(self.model_set), len(self.seg_set))
            assert np.all(np.isfinite(self.scores.data))

        if self.score_mask is None:
            self.score_mask = sparse.csr_matrix(
                np.ones((len(self.model_set), len(self.seg_set)), dtype="bool")
            )
        else:
            assert self.score_mask.shape == (len(self.model_set), len(self.seg_set))

    def filter(self, model_set, seg_set, keep=True, raise_missing=True):
        """Removes elements from TrialScores object.

        Args:
          model_set: List of models to keep or remove.
          seg_set: List of test segments to keep or remove.
          keep: If True, we keep the elements in model_set/seg_set,
                if False, we remove the elements in model_set/seg_set.
          raise_missing: Raises exception if there are elements in model_set or
                         seg_set that are not in the object.
        Returns:
          Filtered TrialScores object.
        """

        if not (keep):
            model_set = np.setdiff1d(self.model_set, model_set)
            seg_set = np.setdiff1d(self.model_set, seg_set)

        f_mod, mod_idx = ismember(model_set, self.model_set)
        f_seg, seg_idx = ismember(seg_set, self.seg_set)

        if not (np.all(f_mod) and np.all(f_seg)):
            for i in (f_mod == 0).nonzero()[0]:
                logging.info("model %s not found" % model_set[i])
            for i in (f_seg == 0).nonzero()[0]:
                logging.info("segment %s not found" % seg_set[i])
            if raise_missing:
                raise Exception("some scores were not computed")

        # model_set = self.model_set[mod_idx]
        # set_set = self.seg_set[seg_idx]
        # ix = np.ix_(mod_idx, seg_idx)

        # logging.info('hola1')
        # new_src = [[self.scores[r,c], i, j] for i,r in enumerate(mod_idx) for j,c in enumerate(seg_idx) if self.score_mask[r,c]]
        # logging.info('hola2')
        # new_data = np.array([r[0] for r in new_src], dtype=float_cpu())
        # new_row = np.array([r[1] for r in new_src], dtype=np.int)
        # new_col = np.array([r[2] for r in new_src], dtype=np.int)
        # logging.info('hola3')
        # shape = (len(model_set), len(seg_set))
        # scores = sparse.coo_matrix((new_data, (new_row, new_col)), shape=shape).tocsr()
        # score_mask = sparse.coo_matrix((np.ones(new_data.shape, dtype=np.bool), (new_row, new_col)), shape=shape).tocsr()

        num_mod = len(model_set)
        num_seg = len(seg_set)
        shape = (num_mod, num_seg)
        scores = self.scores.tocoo()
        new_data = scores.data
        new_row = scores.row.copy()
        for i, r in enumerate(mod_idx):
            if f_mod[i] and i != r:
                idx = scores.row == r
                new_row[idx] = i

        new_col = scores.col.copy()
        for j, c in enumerate(seg_idx):
            if f_seg[j] and j != c:
                idx = scores.col == c
                new_col[idx] = j

        idx = np.logical_and(new_row < num_mod, new_col < num_seg)
        if not np.all(idx):
            new_data = new_data[idx]
            new_row = new_row[idx]
            new_col = new_col[idx]

        scores = sparse.coo_matrix((new_data, (new_row, new_col)), shape=shape).tocsr()

        score_mask = self.score_mask.tocoo()
        new_data = score_mask.data
        new_row = score_mask.row.copy()
        for i, r in enumerate(mod_idx):
            if f_mod[i] and i != r:
                idx = score_mask.row == r
                new_row[idx] = i

        new_col = score_mask.col.copy()
        for j, c in enumerate(seg_idx):
            if f_seg[j] and j != c:
                idx = score_mask.col == c
                new_col[idx] = j

        idx = np.logical_and(new_row < num_mod, new_col < num_seg)
        if not np.all(idx):
            new_data = new_data[idx]
            new_row = new_row[idx]
            new_col = new_col[idx]

        score_mask = sparse.coo_matrix(
            (new_data, (new_row, new_col)), shape=shape
        ).tocsr()

        return SparseTrialScores(model_set, seg_set, scores, score_mask)

    def align_with_ndx(self, ndx, raise_missing=True):
        """Aligns scores, model_set and seg_set with TrialNdx or TrialKey.

        Args:
          ndx: TrialNdx or TrialKey object.
          raise_missing: Raises exception if there are trials in ndx that are not
                         in the score object.

        Returns:
          Aligned TrialScores object.
        """
        scr = self.filter(
            ndx.model_set, ndx.seg_set, keep=True, raise_missing=raise_missing
        )
        if isinstance(ndx, TrialNdx):
            mask = sparse.csr_matrix(ndx.trial_mask)
        elif isinstance(ndx, SparseTrialKey):
            mask = ndx.tar.maximum(ndx.non)
        elif isinstance(ndx, TrialKey):
            mask = sparse.csr_matrix(np.logical_or(ndx.tar, ndx.non))
        else:
            raise Exception()

        mask.eliminate_zeros()
        scr.score_mask = mask.multiply(scr.score_mask)

        mask = mask.tocoo()
        missing_scores = False
        for d, r, c in zip(mask.data, mask.row, mask.col):
            if not scr.score_mask[r, c]:
                missing_scores = True
                logging.info(
                    "missing-scores for %s %s" % (scr.model_set[r], scr.seg_set[c])
                )

        if missing_scores and raise_missing:
            raise Exception("some scores were not computed")

        return scr

    def get_tar_non(self, key):
        """Returns target and non target scores.

        Args:
          key: TrialKey object.

        Returns:
          Numpy array with target scores.
          Numpy array with non-target scores.
        """
        scr = self.align_with_ndx(key)
        tar_mask = scr.score_mask.multiply(key.tar)
        tar = np.array(scr.scores[tar_mask])[0]
        non_mask = scr.score_mask.multiply(key.non)
        non = np.array(scr.scores[non_mask])[0]
        return tar, non

    @classmethod
    def from_trial_scores(cls, scr):
        scores = sparse.csr_matrix(scr.scores)
        score_mask = sparse.csr_matrix(scr.score_mask)
        scores.eliminate_zeros()
        score_mask.eliminate_zeros()
        return cls(scr.model_set, scr.seg_set, scores, score_mask)

    def set_missing_to_value(self, ndx, val):
        """Aligns the scores with a TrialNdx and sets the trials with missing
        scores to the same value.

        Args:
          ndx: TrialNdx or TrialKey object.
          val: Value for the missing scores.

        Returns:
          Aligned SparseTrialScores object.
        """
        scr = self.align_with_ndx(ndx, raise_missing=False)
        if isinstance(ndx, TrialNdx):
            mask = sparse.csr_matrix(ndx.trial_mask)
        elif isinstance(ndx, SparseTrialKey):
            mask = ndx.tar.maximum(ndx.non)
        elif isinstance(ndx, TrialKey):
            mask = sparse.csr_matrix(np.logical_or(ndx.tar, ndx.non))
        else:
            raise Exception()

        mask.eliminate_zeros()
        mask_coo = mask.tocoo()
        for r, c in zip(mask_coo.row, mask_coo.col):
            if not scr.score_mask[r, c]:
                scr.scores[r, c] = val

        scr.score_mask = mask
        return scr

    def __eq__(self, other):
        """Equal operator"""
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(np.isclose(self.scores.data, other.scores.data, atol=1e-5))
        eq = eq and np.all(self.scores.indices == other.scores.indices)
        eq = eq and np.all(self.score_mask.data == other.score_mask.data)
        eq = eq and np.all(self.score_mask.indices == other.score_mask.indices)
        return eq
