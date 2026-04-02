"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from ..hyp_defs import float_cpu
from ..utils.misc import PathLike, build_class_labels_from_boolean_matrix_dense

# from .list_utils import *
from .list_utils import intersect, ismember, list2ndarray, sort, split_list
from .trial_key import TrialKey
from .trial_ndx import TrialNdx


class TrialScores:
    """
    Container for speaker recognition trial scores, compatible with BOSARIS toolkit.

    Attributes:
        model_set (np.ndarray): Array of model IDs.
        seg_set (np.ndarray): Array of segment IDs.
        scores (np.ndarray): Score matrix (num_models x num_segments).
        score_mask (np.ndarray): Boolean matrix indicating which scores are valid.
        q_measures (Optional[Dict[str, np.ndarray]]): Optional dictionary of quality measures.

    Examples:
        >>> import numpy as np
        >>> from hyperion.utils.trial_key import TrialKey
        >>> from hyperion.utils.trial_scores import TrialScores
        >>> key = TrialKey(
        ...     model_set=["m1", "m2"],
        ...     seg_set=["s1", "s2"],
        ...     tar=np.array([[1, 0], [0, 1]], dtype=bool),
        ...     non=np.array([[0, 1], [1, 0]], dtype=bool),
        ... )
        >>> scores = TrialScores(
        ...     model_set=["m1", "m2"],
        ...     seg_set=["s1", "s2"],
        ...     scores=np.array([[2.1, -0.4], [-1.2, 1.7]], dtype=np.float32),
        ...     score_mask=np.ones((2, 2), dtype=bool),
        ... )
        >>> tar, non = scores.get_tar_non(key)
        >>> tar.shape, non.shape
        ((2,), (2,))
    """

    def __init__(
        self,
        model_set: Optional[Union[np.ndarray, List[str]]] = None,
        seg_set: Optional[Union[np.ndarray, List[str]]] = None,
        scores: Optional[np.ndarray] = None,
        score_mask: Optional[np.ndarray] = None,
        q_measures: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.model_set = model_set
        self.seg_set = seg_set
        self.scores = scores
        self.score_mask = score_mask
        self.q_measures = q_measures
        if (model_set is not None) and (seg_set is not None):
            self.validate()

    @property
    def num_models(self) -> int:
        return len(self.model_set)

    @property
    def num_tests(self) -> int:
        return len(self.seg_set)

    def copy(self) -> "TrialScores":
        """Makes a copy of the object"""
        return copy.deepcopy(self)

    def sort(self) -> None:
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        ix = np.ix_(m_idx, s_idx)
        self.scores = self.scores[ix]
        self.score_mask = self.score_mask[ix]
        if self.q_measures is not None:
            for k in self.q_measures.keys():
                self.q_measures[k] = self.q_measures[k][ix]

    def save(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves the object to a file (HDF5, TXT, or CSV/TSV)

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

    def save_h5(self, file_path: PathLike) -> None:
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

    def save_txt(self, file_path: PathLike) -> None:
        """Saves the object to a plain text file (space-separated)

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

    def save_table(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves the object to a CSV/TSV table using Pandas.
        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        output_dir = file_path.parent
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

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
    def load(cls, file_path: PathLike, sep: Optional[str] = None) -> "TrialScores":
        """Loads a TrialScores object from file.
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
    def load_h5(cls, file_path: PathLike) -> "TrialScores":
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
                q_measures = {
                    k: np.asarray(q_grp[k], dtype=float_cpu()) for k in q_grp
                }
            else:
                q_measures = None
        return cls(model_set, seg_set, scores, score_mask, q_measures)

    @classmethod
    def load_txt(cls, file_path: PathLike) -> "TrialScores":
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          TrialScores object.
        """
        rows = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) == 0:
                    continue
                if len(parts) < 3:
                    raise ValueError(
                        f"Malformed line {line_num} in scores file: expected at least 3 columns"
                    )
                rows.append((parts[0], parts[1], parts[2], line_num))

        models = [r[0] for r in rows]
        segments = [r[1] for r in rows]
        scores_v = np.zeros(len(rows), dtype=float_cpu())
        for i, r in enumerate(rows):
            try:
                scores_v[i] = float(r[2])
            except ValueError as e:
                raise ValueError(
                    f"Invalid score value '{r[2]}' at line {r[3]} in scores file"
                ) from e

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
    def load_table(
        cls, file_path: PathLike, sep: Optional[str] = None
    ) -> "TrialScores":
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

        df = pd.read_csv(file_path, sep=sep, dtype={"modelid": str, "segmentid": str})
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
    def merge(cls, scr_list: List["TrialScores"]) -> "TrialScores":
        """Merges several score objects.

        Args:
          scr_list: List of TrialScores objects.

        Returns:
          Merged TrialScores object.
        """
        if len(scr_list) == 0:
            raise ValueError("scr_list must contain at least one TrialScores")
        if len(scr_list) == 1:
            return scr_list[0].copy()

        has_q = [s.q_measures is not None for s in scr_list]
        if any(has_q) and not all(has_q):
            raise ValueError("Cannot merge TrialScores with mixed q_measures presence")

        if all(has_q):
            q_keys = list(scr_list[0].q_measures.keys())
            q_key_set = set(q_keys)
            for s in scr_list[1:]:
                if set(s.q_measures.keys()) != q_key_set:
                    raise ValueError(
                        "All TrialScores must have identical q_measures keys"
                    )
        else:
            q_keys = []

        num_scr = len(scr_list)
        model_set = scr_list[0].model_set
        seg_set = scr_list[0].seg_set
        scores = scr_list[0].scores
        score_mask = scr_list[0].score_mask
        q_measures = (
            {k: scr_list[0].q_measures[k] for k in q_keys} if q_keys else None
        )
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
            scores_1 = np.zeros(shape, dtype=scores.dtype)
            scores_1[ix_a] = scores[ix_b]
            score_mask_1 = np.zeros(shape, dtype="bool")
            score_mask_1[ix_a] = score_mask[ix_b]
            if q_keys:
                q_measures_1 = {
                    k: np.zeros(shape, dtype=q_measures[k].dtype) for k in q_keys
                }
                for k in q_keys:
                    q_measures_1[k][ix_a] = q_measures[k][ix_b]

            _, mi_a, mi_b = intersect(
                new_model_set, scr_i.model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, scr_i.seg_set, assume_unique=True, return_index=True
            )
            ix_a = np.ix_(mi_a, si_a)
            ix_b = np.ix_(mi_b, si_b)
            scores_2 = np.zeros(shape, dtype=scr_i.scores.dtype)
            scores_2[ix_a] = scr_i.scores[ix_b]
            score_mask_2 = np.zeros(shape, dtype="bool")
            score_mask_2[ix_a] = scr_i.score_mask[ix_b]
            if q_keys:
                q_measures_2 = {
                    k: np.zeros(shape, dtype=scr_i.q_measures[k].dtype)
                    for k in q_keys
                }
                for k in q_keys:
                    q_measures_2[k][ix_a] = scr_i.q_measures[k][ix_b]

            model_set = new_model_set
            seg_set = new_seg_set
            scores = scores_1 + scores_2
            if np.any(np.logical_and(score_mask_1, score_mask_2)):
                raise ValueError(
                    "Cannot merge TrialScores with overlapping valid trials"
                )
            score_mask = np.logical_or(score_mask_1, score_mask_2)
            if q_keys:
                for k in q_keys:
                    q_measures[k] = q_measures_1[k] + q_measures_2[k]

        return cls(model_set, seg_set, scores, score_mask, q_measures)

    def filter(
        self,
        model_set: Union[np.ndarray, List[str]],
        seg_set: Union[np.ndarray, List[str]],
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "TrialScores":
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
            seg_set = np.setdiff1d(self.seg_set, seg_set)

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
                raise ValueError("some scores were not computed")

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

    def split(
        self, model_idx: int, num_model_parts: int, seg_idx: int, num_seg_parts: int
    ) -> "TrialScores":
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

    def validate(self) -> None:
        """Validates the attributes of the TrialScores object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        if len(np.unique(self.model_set)) != len(self.model_set):
            raise ValueError("model_set must contain unique entries")
        if len(np.unique(self.seg_set)) != len(self.seg_set):
            raise ValueError("seg_set must contain unique entries")
        if self.scores is None:
            self.scores = np.zeros((len(self.model_set), len(self.seg_set)))
        else:
            expected_shape = (len(self.model_set), len(self.seg_set))
            if self.scores.shape != expected_shape:
                raise ValueError(
                    f"scores shape {self.scores.shape} does not match {expected_shape}"
                )
            if not np.all(np.isfinite(self.scores)):
                raise ValueError("scores must contain only finite values")

        if self.score_mask is None:
            self.score_mask = np.ones(
                (len(self.model_set), len(self.seg_set)), dtype="bool"
            )
        else:
            expected_shape = (len(self.model_set), len(self.seg_set))
            if self.score_mask.shape != expected_shape:
                raise ValueError(
                    f"score_mask shape {self.score_mask.shape} does not match {expected_shape}"
                )

        if self.q_measures is not None:
            for k in self.q_measures.keys():
                if self.q_measures[k].shape != self.scores.shape:
                    raise ValueError(
                        f"q_measures['{k}'] shape {self.q_measures[k].shape} "
                        f"does not match scores shape {self.scores.shape}"
                    )

    def align_with_ndx(
        self, ndx: Union[TrialNdx, TrialKey], raise_missing: bool = True
    ) -> "TrialScores":
        """
        Aligns scores, model_set, and seg_set with a TrialNdx or TrialKey object.

        Args:
            ndx (TrialNdx or TrialKey): Index object indicating which trials to align with.
            raise_missing (bool): Whether to raise an error if some trials are missing.

        Returns:
            TrialScores: Aligned TrialScores object.
        """
        scr = self.filter(
            ndx.model_set, ndx.seg_set, keep=True, raise_missing=raise_missing
        )
        if isinstance(ndx, TrialNdx):
            mask = ndx.trial_mask
        else:
            mask = np.logical_or(ndx.tar, ndx.non)
            # Added to handle ASVSpoof 2024
            if ndx.spoof is not None:
                mask = np.logical_or(mask, ndx.spoof)
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
                raise ValueError("some scores were not computed")
        return scr

    def get_tar_non(self, key: TrialKey) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns target and non-target scores using a TrialKey.

        Args:
            key (TrialKey): TrialKey with target/non-target trial masks.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Target scores, Non-target scores.
        """
        scr = self.align_with_ndx(key)
        tar_mask = np.logical_and(scr.score_mask, key.tar)
        tar = scr.scores[tar_mask]
        non_mask = np.logical_and(scr.score_mask, key.non)
        non = scr.scores[non_mask]
        return tar, non

    def get_tar_non_spoof(
        self, key: TrialKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns target, non-target, and spoofing scores using a TrialKey.

        Args:
            key (TrialKey): TrialKey with target, non-target, and optionally spoof trial masks.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Target scores, Non-target scores, Spoof scores.
        """
        scr = self.align_with_ndx(key)
        tar_mask = np.logical_and(scr.score_mask, key.tar)
        tar = scr.scores[tar_mask]
        non_mask = np.logical_and(scr.score_mask, key.non)
        non = scr.scores[non_mask]
        if key.spoof is None:
            spoof = np.empty((0,), dtype=scr.scores.dtype)
        else:
            spoof_mask = np.logical_and(scr.score_mask, key.spoof)
            spoof = scr.scores[spoof_mask]
        return tar, non, spoof

    def get_tar_non_q_measures(
        self,
        key: TrialKey,
        q_names: Optional[List[str]] = None,
        return_dict: bool = False,
    ) -> Tuple[
        Union[Dict[str, np.ndarray], np.ndarray],
        Union[Dict[str, np.ndarray], np.ndarray],
    ]:
        """
        Returns quality measures for target and non-target trials.

        Args:
            key (TrialKey): TrialKey object.
            q_names (list of str, optional): Names of quality measures to extract. All are used if None.
            return_dict (bool): If True, returns dictionaries; if False, returns stacked arrays.

        Returns:
            Tuple: (target quality measures, non-target quality measures)
        """
        scr = self.align_with_ndx(key)
        if scr.q_measures is None:
            raise ValueError("q_measures are not available in TrialScores")

        tar_mask = np.logical_and(scr.score_mask, key.tar)
        if q_names is None:
            q_names = list(scr.q_measures.keys())
        else:
            missing_q = [q for q in q_names if q not in scr.q_measures]
            if missing_q:
                raise ValueError(
                    f"Requested q_names not found in q_measures: {missing_q}"
                )
        tar = {}
        for k in q_names:
            tar[k] = scr.q_measures[k][tar_mask]
        non_mask = np.logical_and(scr.score_mask, key.non)
        non = {}
        for k in q_names:
            non[k] = scr.q_measures[k][non_mask]

        if not return_dict:
            if len(q_names) == 0:
                tar = np.empty((int(np.sum(tar_mask)), 0), dtype=float_cpu())
                non = np.empty((int(np.sum(non_mask)), 0), dtype=float_cpu())
                return tar, non
            tar = np.vstack(tuple(tar[k] for k in q_names)).T
            non = np.vstack(tuple(non[k] for k in q_names)).T
        return tar, non

    def get_class_sim(
        self,
        key: TrialKey,
        model_classes: Union[List[str], np.ndarray, None] = None,
        seg_classes: Union[List[str], np.ndarray, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the class similarity scores for the trials in key.

        Args:
          key: SparseTrialKey object.

        Returns:
          Numpy array with the class similarity scores.
          M(i,j) average similarity between class i and class j.
        """
        scr = self.align_with_ndx(key)
        tar_mask = np.logical_and(scr.score_mask, key.tar)
        non_mask = np.logical_and(scr.score_mask, key.non)
        score_mask = np.logical_or(tar_mask, non_mask).astype(float)

        if model_classes is None or seg_classes is None:
            logging.info(
                "model/seg classes not provided, building from key.tar, it can take a while"
            )
            model_classes, seg_classes = build_class_labels_from_boolean_matrix_dense(
                key.tar
            )

        unique_model_classes = np.unique(model_classes)
        unique_seg_classes = np.unique(seg_classes)
        sim_matrix = np.zeros(
            (len(unique_model_classes), len(unique_seg_classes)),
            dtype=self.scores.dtype,
        )
        for i, rc in enumerate(unique_model_classes):
            row_mask = model_classes == rc
            for j, cc in enumerate(unique_seg_classes):
                col_mask = seg_classes == cc
                idx = np.ix_(row_mask, col_mask)
                block = scr.scores[idx] * score_mask[idx]
                count = np.sum(score_mask[idx])
                sim_matrix[i, j] = (
                    block.sum() / count if block.size > 0 and count > 0 else np.nan
                )

        return sim_matrix, unique_model_classes, unique_seg_classes

    def set_missing_to_value(
        self, ndx: Union[TrialNdx, TrialKey], val: float
    ) -> "TrialScores":
        """
        Sets scores missing in `score_mask` but present in `ndx` to a specific value.

        Args:
            ndx (TrialNdx or TrialKey): Index of trials.
            val (float): Value to assign to missing scores.

        Returns:
            TrialScores: The modified TrialScores object.
        """
        scr = self.align_with_ndx(ndx, raise_missing=False)
        if isinstance(ndx, TrialNdx):
            mask = ndx.trial_mask
        else:
            mask = np.logical_or(ndx.tar, ndx.non)
            if ndx.spoof is not None:
                mask = np.logical_or(mask, ndx.spoof)
        mask = np.logical_and(np.logical_not(scr.score_mask), mask)
        scr.scores[mask] = val
        scr.score_mask[mask] = True
        return scr

    def transform(self, f: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Applies a transformation function to the scores at valid (True) score_mask positions.

        Args:
            f (callable): A function to apply to score values.
        """
        mask = self.score_mask
        self.scores[mask] = f(self.scores[mask])

    def __eq__(self, other: object) -> bool:
        """Equal operator"""
        if not isinstance(other, TrialScores):
            return False
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(np.isclose(self.scores, other.scores, atol=1e-5))
        eq = eq and np.all(self.score_mask == other.score_mask)
        eq = eq and ((self.q_measures is None) == (other.q_measures is None))
        if eq and self.q_measures is not None and other.q_measures is not None:
            eq = self.q_measures.keys() == other.q_measures.keys()
            if eq:
                for k in self.q_measures.keys():
                    eq = eq and np.all(
                        np.isclose(self.q_measures[k], other.q_measures[k], atol=1e-5)
                    )

        return eq

    def __ne__(self, other: object) -> bool:
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other: object) -> int:
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1

    def test(key_file: PathLike = "core-core_det5_key.h5") -> None:
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
