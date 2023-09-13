"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..hyp_defs import float_cpu

# from .list_utils import *
from .list_utils import intersect, ismember, list2ndarray, sort, split_list
from .trial_key import TrialKey
from .trial_ndx import TrialNdx


class TrialScores(object):
    """Contains the scores for the speaker recognition trials.
        Bosaris compatible Scores.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      scores: Matrix with the scores (num_models x num_segments).
      score_mask: Boolean matrix with the trials with valid scores to True (num_models x num_segments).
      q_measures: optional dictionary of quality measure matrices
    """

    def __init__(
        self,
        model_set=None,
        seg_set=None,
        scores=None,
        score_mask=None,
        q_measures=None,
    ):
        self.model_set = model_set
        self.seg_set = seg_set
        self.scores = scores
        self.score_mask = score_mask
        self.q_measures = q_measures
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
        self.scores = self.scores[ix]
        self.score_mask = self.score_mask[ix]
        if self.q_measures is not None:
            for k in self.q_measures.keys():
                self.q_measures[k] = self.q_measures[k][ix]

    def save(self, file_path, sep=None):
        """Saves object to txt/h5 file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix
        if file_ext in [".h5", ".hdf5"]:
            self.save_h5(file_path)
        elif file_ext in ["", ".txt"]:
            self.save_txt(file_path)
        else:
            self.save_table(file_path, sep=sep)

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
            f.create_dataset("scores", data=self.scores)
            f.create_dataset("score_mask", data=self.score_mask.astype("uint8"))
            if self.q_measures is not None:
                q_grp = f.create_group("q_measures")
                for k, v in self.q_measures.items():
                    q_grp.create_dataset(k, data=v)

    def save_txt(self, file_path):
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        idx = (self.score_mask.T == True).nonzero()
        with open(file_path, "w") as f:
            for item in zip(idx[0], idx[1]):
                f.write(
                    "%s %s %f\n"
                    % (
                        self.model_set[item[1]],
                        self.seg_set[item[0]],
                        self.scores[item[1], item[0]],
                    )
                )

        if self.q_measures is not None:
            logging.warning("q_measures cannot be saved to txt file")

    def save_table(self, file_path, sep=None):
        """Saves object to pandas tabnle file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        q_str = ""
        if self.q_measures is not None:
            q_str = sep + sep.join(self.q_measures.keys())

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"modelid{sep}segmentid{sep}LLR{q_str}\n")
            I, J = self.score_mask.nonzero()
            for i, j in zip(I, J):
                if self.q_measures is not None:
                    q_str = sep + sep.join(
                        [str(v[i, j]) for k, v in self.q_measures.items()]
                    )
                f.write(
                    f"{self.model_set[i]}{sep}{self.seg_set[j]}{sep}{self.scores[i,j]}{q_str}\n"
                )

    @classmethod
    def load(cls, file_path, sep=None):
        """Loads object from txt/h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialScores object.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix
        if file_ext in (".h5", ".hdf5"):
            return cls.load_h5(file_path)
        elif file_ext in ("", ".txt"):
            return cls.load_txt(file_path)
        else:
            return cls.load_table(file_path, sep)

    @classmethod
    def load_h5(cls, file_path):
        """Loads object from h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialScores object.
        """
        with h5py.File(file_path, "r") as f:
            model_set = [t.decode("utf-8") for t in f["ID/row_ids"]]
            seg_set = [t.decode("utf-8") for t in f["ID/column_ids"]]
            scores = np.asarray(f["scores"], dtype=float_cpu())
            score_mask = np.asarray(f["score_mask"], dtype="bool")
            if "q_measures" in f:
                q_grp = f["q_measures"]
                q_measures = {k: q_grp[k] for k in q_grp}
            else:
                q_measures = None
        return cls(model_set, seg_set, scores, score_mask, q_measures)

    @classmethod
    def load_txt(cls, file_path):
        """Loads object from h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialScores object.
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

        scores = np.zeros((len(model_set), len(seg_set)))
        score_mask = np.zeros(scores.shape, dtype="bool")
        for item in zip(model_idx, seg_idx, scores_v):
            score_mask[item[0], item[1]] = True
            scores[item[0], item[1]] = item[2]
        return cls(model_set, seg_set, scores, score_mask)

    @classmethod
    def load_table(cls, file_path, sep=None):
        """Loads object from pandas table file

        Args:
          file_path: File to read the list.

        Returns:
          TrialScores object.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        df = pd.read_csv(file_path, sep=sep)
        models = df["modelid"].values
        segments = df["segmentid"].values
        score_list = df["LLR"].values
        model_set, model_idx = np.unique(models, return_inverse=True)
        seg_set, seg_idx = np.unique(segments, return_inverse=True)
        score_mask = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        scores = np.zeros((len(model_set), len(seg_set)), dtype=float_cpu())
        for i, j, score in zip(model_idx, seg_idx, score_list):
            score_mask[i, j] = True
            scores[i, j] = score

        if len(df.columns) > 3:
            q_names = df.columns[3:]
            q_vals = df.iloc[:, 3:].values
            q_measures = {}
            for q_name in q_names:
                q_measures[q_name] = np.zeros(scores.shape, dtype=float_cpu())

            for i, j, q_row in zip(model_idx, seg_idx, q_vals):
                for col, q_name in enumerate(q_names):
                    q_measures[q_name][i, j] = q_row[col]

        else:
            q_measures = None

        return cls(model_set, seg_set, scores, score_mask, q_measures)

    @classmethod
    def merge(cls, scr_list):
        """Merges several score objects.

        Args:
          scr_list: List of TrialNdx objects.

        Returns:
          Merged TrialScores object.
        """
        num_scr = len(scr_list)
        model_set = scr_list[0].model_set
        seg_set = scr_list[0].seg_set
        scores = scr_list[0].scores
        score_mask = scr_list[0].score_mask
        q_measures = scr_list[0].q_measures
        for i in range(1, num_scr):
            scr_i = scr_list[i]
            new_model_set = np.union1d(model_set, scr_i.model_set)
            new_seg_set = np.union1d(seg_set, scr_i.seg_set)
            shape = (len(new_model_set), len(new_seg_set))

            _, mi_a, mi_b = intersect(
                new_model_set, model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, seg_set, assume_unique=True, return_index=True
            )
            ix_a = np.ix_(mi_a, si_a)
            ix_b = np.ix_(mi_b, si_b)
            scores_1 = np.zeros(shape)
            scores_1[ix_a] = scores[ix_b]
            score_mask_1 = np.zeros(shape, dtype="bool")
            score_mask_1[ix_a] = score_mask[ix_b]
            if q_measures is not None:
                q_measures_1 = {k: np.zeros(shape) for k in q_measures.keys()}
                for k in q_measures.keys():
                    q_measures_1[k][ix_a] = q_measures[k][ix_b]

            trial_mask_2 = np.zeros(
                (len(new_model_set), len(new_seg_set)), dtype="bool"
            )
            _, mi_a, mi_b = intersect(
                new_model_set, scr_i.model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, scr_i.seg_set, assume_unique=True, return_index=True
            )
            ix_a = np.ix_(mi_a, si_a)
            ix_b = np.ix_(mi_b, si_b)
            scores_2 = np.zeros(shape)
            scores_2[ix_a] = scr_i.scores[ix_b]
            score_mask_2 = np.zeros(shape, dtype="bool")
            score_mask_2[ix_a] = scr_i.score_mask[ix_b]
            if q_measures is not None:
                q_measures_2 = {k: np.zeros(shape) for k in q_measures.keys()}
                for k in q_measures.keys():
                    q_measures_2[k][ix_a] = scr_i.q_measures[k][ix_b]

            model_set = new_model_set
            seg_set = new_seg_set
            scores = scores_1 + scores_2
            assert not (np.any(np.logical_and(score_mask_1, score_mask_2)))
            score_mask = np.logical_or(score_mask_1, score_mask_2)
            if q_measures is not None:
                for k in q_measures.keys():
                    q_measures[k] = q_measures_1[k] + q_measures_2[k]

        return cls(model_set, seg_set, scores, score_mask, q_measures)

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

        if not keep:
            model_set = np.setdiff1d(self.model_set, model_set)
            seg_set = np.setdiff1d(self.model_set, seg_set)

        f_mod, mod_idx = ismember(model_set, self.model_set)
        f_seg, seg_idx = ismember(seg_set, self.seg_set)
        q_measures = None
        if np.all(f_mod) and np.all(f_seg):
            model_set = self.model_set[mod_idx]
            seg_set = self.seg_set[seg_idx]
            ix = np.ix_(mod_idx, seg_idx)
            scores = self.scores[ix]
            score_mask = self.score_mask[ix]
            if self.q_measures is not None:
                q_measures = {}
                for k in self.q_measures.keys():
                    q_measures[k] = self.q_measures[k][ix]
        else:
            for i in (f_mod == 0).nonzero()[0]:
                logging.info("model %s not found", model_set[i])
            for i in (f_seg == 0).nonzero()[0]:
                logging.info("segment %s not found", seg_set[i])
            if raise_missing:
                raise Exception("some scores were not computed")

            scores = np.zeros((len(model_set), len(seg_set)), dtype=float_cpu())
            score_mask = np.zeros(scores.shape, dtype=bool)
            ix1 = np.ix_(f_mod, f_seg)
            ix2 = np.ix_(mod_idx[f_mod], seg_idx[f_seg])
            scores[ix1] = self.scores[ix2]
            score_mask[ix1] = self.score_mask[ix2]
            if self.q_measures is not None:
                q_measures = {}
                for k in self.q_measures.keys():
                    q_measures[k] = np.zeros(scores.shape, dtype=float_cpu())
                    q_measures[k][ix1] = self.q_measures[k][ix2]

        return TrialScores(model_set, seg_set, scores, score_mask, q_measures)

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
        q_measures = None
        if self.q_measures is not None:
            q_measures = {}
            for k in self.q_measures.keys():
                q_measures[k] = self.q_measures[k][ix]

        return TrialScores(model_set, seg_set, scores, score_mask, q_measures)

    def validate(self):
        """Validates the attributes of the TrialScores object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        assert len(np.unique(self.model_set)) == len(self.model_set)
        assert len(np.unique(self.seg_set)) == len(self.seg_set)
        if self.scores is None:
            self.scores = np.zeros((len(self.model_set), len(self.seg_set)))
        else:
            assert self.scores.shape == (len(self.model_set), len(self.seg_set))
            assert np.all(np.isfinite(self.scores))

        if self.score_mask is None:
            self.score_mask = np.ones(
                (len(self.model_set), len(self.seg_set)), dtype="bool"
            )
        else:
            assert self.score_mask.shape == (len(self.model_set), len(self.seg_set))

        if self.q_measures is not None:
            for k in self.q_measures.keys():
                assert self.q_measures[k].shape == self.scores.shape

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
            mask = ndx.trial_mask
        else:
            mask = np.logical_or(ndx.tar, ndx.non)
        scr.score_mask = np.logical_and(mask, scr.score_mask)

        missing_trials = np.logical_and(mask, np.logical_not(scr.score_mask))
        missing = np.any(missing_trials)
        if missing:
            idx = (missing_trials == True).nonzero()
            for i, j in zip(idx[0], idx[1]):
                logging.info(
                    "missing-scores for %s %s" % (scr.model_set[i], scr.seg_set[j])
                )

            if raise_missing:
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
        tar_mask = np.logical_and(scr.score_mask, key.tar)
        tar = scr.scores[tar_mask]
        non_mask = np.logical_and(scr.score_mask, key.non)
        non = scr.scores[non_mask]
        return tar, non

    def get_tar_non_q_measures(self, key, q_names=None, return_dict=False):
        """Returns target and non target scores.

        Args:
          key: TrialKey object.
          q_names: names of quality measures to return, if None it will return all

        Returns:
          Numpy array with target scores.
          Numpy array with non-target scores.
        """
        scr = self.align_with_ndx(key)
        tar_mask = np.logical_and(scr.score_mask, key.tar)
        if q_names is None:
            q_names = self.q_measures.keys()
        tar = {}
        for k in q_names:
            tar[k] = self.q_measures[k][tar_mask]
        non_mask = np.logical_and(scr.score_mask, key.non)
        non = {}
        for k in q_names:
            non[k] = self.q_measures[k][non_mask]

        if not return_dict:
            tar = np.vstack(tuple(tar[k] for k in q_names)).T
            non = np.vstack(tuple(non[k] for k in q_names)).T
        return tar, non

    def set_missing_to_value(self, ndx, val):
        """Aligns the scores with a TrialNdx and sets the trials with missing
        scores to the same value.

        Args:
          ndx: TrialNdx or TrialKey object.
          val: Value for the missing scores.

        Returns:
          Aligned TrialScores object.
        """
        scr = self.align_with_ndx(ndx, raise_missing=False)
        if isinstance(ndx, TrialNdx):
            mask = ndx.trial_mask
        else:
            mask = np.logical_or(ndx.tar, ndx.non)
        mask = np.logical_and(np.logical_not(scr.score_mask), mask)
        scr.scores[mask] = val
        scr.score_mask[mask] = True
        return scr

    def transform(self, f):
        """Applies a function to the valid scores of the object.

        Args:
          f: function handle.
        """
        mask = self.score_mask
        self.scores[mask] = f(self.scores[mask])

    def __eq__(self, other):
        """Equal operator"""
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(np.isclose(self.scores, other.scores, atol=1e-5))
        eq = eq and np.all(self.score_mask == other.score_mask)
        if self.q_measures is not None:
            eq = eq and other.q_measures is not None
            if eq:
                eq = self.q_measures.keys() == other.q_measures.keys()
                if eq:
                    for k in self.q_measures.keys():
                        eq = eq and np.all(
                            np.isclose(
                                self.q_measures[k], other.q_measures[k], atol=1e-5
                            )
                        )

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
        key = TrialKey.load(key_file)

        mask = np.logical_or(key.tar, key.non)
        scr1 = TrialScores(
            key.model_set,
            key.seg_set,
            np.random.normal(size=key.tar.shape) * mask,
            mask,
        )

        scr2 = scr1.copy()
        scr2.sort()
        assert scr2 != scr1
        scr3 = scr2.align_with_ndx(key)
        assert scr1 == scr3

        scr1.sort()
        scr2 = scr1.copy()

        scr2.model_set[0] = "m1"
        scr2.score_mask[:] = 0
        assert np.any(scr1.model_set != scr2.model_set)
        assert np.any(scr1.score_mask != scr2.score_mask)

        scr2 = TrialScores(
            scr1.model_set[:10],
            scr1.seg_set,
            scr1.scores[:10, :],
            scr1.score_mask[:10, :],
        )
        scr3 = TrialScores(
            scr1.model_set[10:],
            scr1.seg_set,
            scr1.scores[10:, :],
            scr1.score_mask[10:, :],
        )
        scr4 = TrialScores.merge([scr2, scr3])
        assert scr1 == scr4

        scr2 = TrialScores(
            scr1.model_set,
            scr1.seg_set[:10],
            scr1.scores[:, :10],
            scr1.score_mask[:, :10],
        )
        scr3 = TrialScores(
            scr1.model_set,
            scr1.seg_set[10:],
            scr1.scores[:, 10:],
            scr1.score_mask[:, 10:],
        )
        scr4 = TrialScores.merge([scr2, scr3])
        assert scr1 == scr4

        scr2 = TrialScores(
            scr1.model_set[:5],
            scr1.seg_set[:10],
            scr1.scores[:5, :10],
            scr1.score_mask[:5, :10],
        )
        scr3 = scr1.filter(scr2.model_set, scr2.seg_set, keep=True)
        assert scr2 == scr3

        num_parts = 3
        scr_list = []
        for i in range(num_parts):
            for j in range(num_parts):
                scr_ij = scr1.split(i + 1, num_parts, j + 1, num_parts)
                scr_list.append(scr_ij)
        scr2 = TrialScores.merge(scr_list)
        assert scr1 == scr2

        f = lambda x: 3 * x + 1
        scr2 = scr1.copy()
        scr2.score_mask[0, 0] = True
        scr2.score_mask[0, 1] = False
        scr4 = scr2.copy()
        scr4.transform(f)
        assert scr4.scores[0, 0] == 3 * scr1.scores[0, 0] + 1
        assert scr4.scores[0, 1] == scr1.scores[0, 1]

        scr2 = scr1.align_with_ndx(key)
        key2 = key.copy()
        scr2.score_mask[:] = False
        scr2.score_mask[0, 0] = True
        scr2.score_mask[0, 1] = True
        scr2.scores[0, 0] = 1
        scr2.scores[0, 1] = -1
        key2.tar[:] = False
        key2.non[:] = False
        key2.tar[0, 0] = True
        key2.non[0, 1] = True
        [tar, non] = scr2.get_tar_non(key2)
        assert np.all(tar == [1])
        assert np.all(non == [-1])

        scr2.score_mask[0, 0] = False
        scr4 = scr2.set_missing_to_value(key2, -10)
        assert scr4.scores[0, 0] == -10

        file_h5 = "test.h5"
        scr1.save(file_h5)
        scr2 = TrialScores.load(file_h5)
        assert scr1 == scr2

        file_txt = "test.txt"
        scr3.score_mask[0, :] = True
        scr3.score_mask[:, 0] = True
        scr3.save(file_txt)
        scr2 = TrialScores.load(file_txt)
        assert scr3 == scr2
