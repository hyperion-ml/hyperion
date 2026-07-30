"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from ..hyp_defs import float_cpu
from .list_utils import *
from .misc import PathLike, build_class_labels_from_boolean_matrix_sparse
from .sparse_trial_key import SparseTrialKey
from .trial_key import TrialKey
from .trial_ndx import TrialNdx
from .trial_scores import TrialScores


class SparseTrialScores(TrialScores):
    """Contains the scores for the speaker recognition trials.
        Bosaris compatible Scores.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      scores: Matrix with the scores (num_models x num_segments).
      score_mask: Boolean matrix with the trials with valid scores to True (num_models x num_segments).

    Examples:
      >>> import numpy as np
      >>> from hyperion.utils.trial_scores import TrialScores
      >>> from hyperion.utils.sparse_trial_scores import SparseTrialScores
      >>> dense = TrialScores(
      ...     model_set=["m1", "m2"],
      ...     seg_set=["s1", "s2"],
      ...     scores=np.array([[1.2, 0.0], [0.0, -0.8]], dtype=np.float32),
      ...     score_mask=np.array([[1, 0], [0, 1]], dtype=bool),
      ... )
      >>> sparse_scores = SparseTrialScores.from_trial_scores(dense)
      >>> sparse_scores.score_mask.nnz
      2
      >>> dense_back = sparse_scores.to_trial_scores()
      >>> dense_back.scores.shape
      (2, 2)
    """

    def __init__(
        self,
        model_set: Optional[Union[np.ndarray, List[str]]] = None,
        seg_set: Optional[Union[np.ndarray, List[str]]] = None,
        scores: Optional[sparse.spmatrix] = None,
        score_mask: Optional[sparse.spmatrix] = None,
    ) -> None:
        super().__init__(model_set, seg_set, scores, score_mask)

    @staticmethod
    def _extract_scores_from_mask(
        scores: sparse.spmatrix, mask: sparse.spmatrix
    ) -> np.ndarray:
        """Extracts scores selected by a sparse mask as a 1-D NumPy array."""
        mask = mask.tocoo()
        if mask.nnz == 0:
            return np.empty((0,), dtype=scores.dtype)
        valid = np.asarray(mask.data, dtype=bool)
        if not np.all(valid):
            row = mask.row[valid]
            col = mask.col[valid]
            if row.size == 0:
                return np.empty((0,), dtype=scores.dtype)
        else:
            row = mask.row
            col = mask.col

        values = scores[row, col]
        if sparse.issparse(values):
            return values.toarray().ravel()
        return np.asarray(values).ravel()

    def sort(self) -> None:
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        scores = self.scores.tocsr()[m_idx][:, s_idx].tocsr()
        score_mask = self.score_mask.tocsr()[m_idx][:, s_idx].tocsr()
        scores.eliminate_zeros()
        score_mask.eliminate_zeros()
        scores.sort_indices()
        score_mask.sort_indices()
        self.scores = scores
        self.score_mask = score_mask

    def save_h5(self, file_path: PathLike) -> None:
        raise NotImplementedError()

    def save_txt(self, file_path: PathLike) -> None:
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

    def save_table(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves object to a pandas table file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        self.score_mask.eliminate_zeros()
        score_mask = self.score_mask.tocoo()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"modelid{sep}segmentid{sep}LLR\n")
            for i, j in zip(score_mask.row, score_mask.col):
                f.write(
                    f"{self.model_set[i]}{sep}{self.seg_set[j]}{sep}{self.scores[i,j]}\n"
                )

    @classmethod
    def load_h5(cls, file_path: PathLike) -> "SparseTrialScores":
        raise NotImplementedError()

    @classmethod
    def load_txt(cls, file_path: PathLike) -> "SparseTrialScores":
        """Loads object from h5 file

        Args:
          file_path: File to read the list.

        Returns:
          SparseTrialScores object.
        """
        models: List[str] = []
        segments: List[str] = []
        scores_v: List[float] = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) == 0:
                    continue
                if len(parts) < 3:
                    raise ValueError(
                        f"Malformed line {line_num} in score file: expected at least 3 columns"
                    )
                models.append(parts[0])
                segments.append(parts[1])
                try:
                    scores_v.append(float(parts[2]))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid score value at line {line_num}: '{parts[2]}'"
                    ) from e

        scores_v = np.asarray(scores_v, dtype=float_cpu())

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
    def load_table(
        cls, file_path: PathLike, sep: Optional[str] = None
    ) -> "SparseTrialScores":
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
        scores = sparse.lil_matrix((len(model_set), len(seg_set)), dtype=float_cpu())
        score_mask = sparse.lil_matrix(scores.shape, dtype="bool")
        for i, j, score in zip(model_idx, seg_idx, score_list):
            score_mask[i, j] = True
            scores[i, j] = score

        return cls(model_set, seg_set, scores.tocsr(), score_mask.tocsr())

    @classmethod
    def merge(cls, scr_list: List["SparseTrialScores"]) -> "SparseTrialScores":
        """Merges several SparseTrialScores objects.

        Args:
          scr_list: List of SparseTrialScores objects.

        Returns:
          Merged SparseTrialScores object.
        """
        if len(scr_list) == 0:
            raise ValueError("scr_list must contain at least one SparseTrialScores")
        if len(scr_list) == 1:
            return scr_list[0].copy()

        model_set = scr_list[0].model_set
        seg_set = scr_list[0].seg_set
        score_mask = scr_list[0].score_mask.tocsr()
        score_mask.eliminate_zeros()
        score_mask.sort_indices()
        scores = scr_list[0].scores.tocsr().multiply(score_mask)
        scores.eliminate_zeros()
        scores.sort_indices()

        for i in range(1, len(scr_list)):
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
            model_map_1 = np.empty(len(model_set), dtype="int64")
            seg_map_1 = np.empty(len(seg_set), dtype="int64")
            model_map_1[mi_b] = mi_a
            seg_map_1[si_b] = si_a

            score_mask_1 = score_mask.tocoo()
            score_mask_1 = sparse.coo_matrix(
                (
                    score_mask_1.data,
                    (model_map_1[score_mask_1.row], seg_map_1[score_mask_1.col]),
                ),
                shape=shape,
                dtype="bool",
            ).tocsr()
            score_mask_1.eliminate_zeros()
            score_mask_1.sort_indices()

            scores_1 = scores.tocoo()
            scores_1 = sparse.coo_matrix(
                (scores_1.data, (model_map_1[scores_1.row], seg_map_1[scores_1.col])),
                shape=shape,
                dtype=scores.dtype,
            ).tocsr()
            scores_1.eliminate_zeros()
            scores_1.sort_indices()

            _, mi_a, mi_b = intersect(
                new_model_set, scr_i.model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, scr_i.seg_set, assume_unique=True, return_index=True
            )
            model_map_2 = np.empty(len(scr_i.model_set), dtype="int64")
            seg_map_2 = np.empty(len(scr_i.seg_set), dtype="int64")
            model_map_2[mi_b] = mi_a
            seg_map_2[si_b] = si_a

            score_mask_i = scr_i.score_mask.tocsr()
            score_mask_i.eliminate_zeros()
            score_mask_i.sort_indices()
            score_mask_2 = score_mask_i.tocoo()
            score_mask_2 = sparse.coo_matrix(
                (
                    score_mask_2.data,
                    (model_map_2[score_mask_2.row], seg_map_2[score_mask_2.col]),
                ),
                shape=shape,
                dtype="bool",
            ).tocsr()
            score_mask_2.eliminate_zeros()
            score_mask_2.sort_indices()

            scores_i = scr_i.scores.tocsr().multiply(score_mask_i)
            scores_i.eliminate_zeros()
            scores_i.sort_indices()
            scores_2 = scores_i.tocoo()
            scores_2 = sparse.coo_matrix(
                (scores_2.data, (model_map_2[scores_2.row], seg_map_2[scores_2.col])),
                shape=shape,
                dtype=scores_i.dtype,
            ).tocsr()
            scores_2.eliminate_zeros()
            scores_2.sort_indices()

            overlap = score_mask_1.multiply(score_mask_2)
            overlap.eliminate_zeros()
            if overlap.nnz > 0:
                raise ValueError(
                    "Cannot merge SparseTrialScores with overlapping valid trials"
                )

            model_set = new_model_set
            seg_set = new_seg_set
            scores = scores_1 + scores_2
            score_mask = score_mask_1.maximum(score_mask_2).tocsr()
            score_mask.eliminate_zeros()
            score_mask.sort_indices()
            scores = scores.multiply(score_mask)
            scores.eliminate_zeros()
            scores.sort_indices()

        return cls(model_set, seg_set, scores, score_mask)

    def split(
        self, model_idx: int, num_model_parts: int, seg_idx: int, num_seg_parts: int
    ) -> "SparseTrialScores":
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

    def validate(self) -> None:
        """Validates the attributes of the TrialKey object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        if len(np.unique(self.model_set)) != len(self.model_set):
            raise ValueError("model_set must contain unique entries")
        if len(np.unique(self.seg_set)) != len(self.seg_set):
            raise ValueError("seg_set must contain unique entries")
        if self.scores is None:
            self.scores = sparse.csr_matrix(
                (len(self.model_set), len(self.seg_set)), dtype=float_cpu()
            )
        else:
            self.scores = self.scores.tocsr()
            expected_shape = (len(self.model_set), len(self.seg_set))
            if self.scores.shape != expected_shape:
                raise ValueError(
                    f"scores shape {self.scores.shape} does not match {expected_shape}"
                )
            if not np.all(np.isfinite(self.scores.data)):
                raise ValueError("scores must contain only finite values")

        if self.score_mask is None:
            self.score_mask = sparse.csr_matrix(
                np.ones((len(self.model_set), len(self.seg_set)), dtype="bool")
            )
        else:
            self.score_mask = self.score_mask.tocsr()
            expected_shape = (len(self.model_set), len(self.seg_set))
            if self.score_mask.shape != expected_shape:
                raise ValueError(
                    f"score_mask shape {self.score_mask.shape} does not match {expected_shape}"
                )

        self.scores.eliminate_zeros()
        self.score_mask.eliminate_zeros()
        self.scores.sort_indices()
        self.score_mask.sort_indices()

    def filter(
        self,
        model_set: Union[np.ndarray, List[str]],
        seg_set: Union[np.ndarray, List[str]],
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "SparseTrialScores":
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
        if np.all(f_mod) and np.all(f_seg):
            model_set = self.model_set[mod_idx]
            seg_set = self.seg_set[seg_idx]
            scores = self.scores.tocsr()[mod_idx][:, seg_idx].tocsr()
            score_mask = self.score_mask.tocsr()[mod_idx][:, seg_idx].tocsr()
        else:
            for i in (f_mod == 0).nonzero()[0]:
                logging.info("model %s not found", model_set[i])
            for i in (f_seg == 0).nonzero()[0]:
                logging.info("segment %s not found", seg_set[i])
            if raise_missing:
                raise ValueError("some scores were not computed")
            shape = (len(model_set), len(seg_set))
            scores = sparse.csr_matrix(shape, dtype=self.scores.dtype)
            score_mask = sparse.csr_matrix(shape, dtype="bool")
            row_pos = np.where(f_mod)[0]
            col_pos = np.where(f_seg)[0]
            if row_pos.size > 0 and col_pos.size > 0:
                src_scores = self.scores.tocsr()[mod_idx[f_mod]][:, seg_idx[f_seg]]
                src_mask = self.score_mask.tocsr()[mod_idx[f_mod]][:, seg_idx[f_seg]]
                scores = scores.tolil()
                score_mask = score_mask.tolil()
                ix = np.ix_(row_pos, col_pos)
                scores[ix] = src_scores
                score_mask[ix] = src_mask
                scores = scores.tocsr()
                score_mask = score_mask.tocsr()

        scores.eliminate_zeros()
        score_mask.eliminate_zeros()
        scores.sort_indices()
        score_mask.sort_indices()
        return SparseTrialScores(model_set, seg_set, scores, score_mask)

    def align_with_ndx(
        self,
        ndx: Union[TrialNdx, TrialKey, SparseTrialKey],
        raise_missing: bool = True,
    ) -> "SparseTrialScores":
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
            if ndx.spoof is not None:
                mask = mask.maximum(ndx.spoof)
        elif isinstance(ndx, TrialKey):
            mask_dense = np.logical_or(ndx.tar, ndx.non)
            if ndx.spoof is not None:
                mask_dense = np.logical_or(mask_dense, ndx.spoof)
            mask = sparse.csr_matrix(mask_dense)
        else:
            raise ValueError(f"Unsupported ndx type: {type(ndx).__name__}")

        mask.eliminate_zeros()
        scr.score_mask = mask.multiply(scr.score_mask)

        mask = mask.tocoo()
        missing_scores = False
        for d, r, c in zip(mask.data, mask.row, mask.col):
            if not scr.score_mask[r, c]:
                missing_scores = True
                logging.info(
                    "missing-scores for %s %s", scr.model_set[r], scr.seg_set[c]
                )

        if missing_scores and raise_missing:
            raise ValueError("some scores were not computed")

        return scr

    def get_tar_non(
        self, key: Union[TrialKey, SparseTrialKey]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns target and non target scores.

        Args:
          key: TrialKey object.

        Returns:
          Numpy array with target scores.
          Numpy array with non-target scores.
        """
        scr = self.align_with_ndx(key)
        tar_mask = scr.score_mask.multiply(key.tar)
        tar = self._extract_scores_from_mask(scr.scores, tar_mask)
        non_mask = scr.score_mask.multiply(key.non)
        non = self._extract_scores_from_mask(scr.scores, non_mask)
        return tar, non

    def get_tar_non_spoof(
        self, key: Union[TrialKey, SparseTrialKey]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns target, non-target and spoof scores.

        Args:
          key: TrialKey or SparseTrialKey object.

        Returns:
          Numpy array with target scores.
          Numpy array with non-target scores.
          Numpy array with spoof scores.
        """
        scr = self.align_with_ndx(key)
        tar_mask = scr.score_mask.multiply(key.tar)
        tar = self._extract_scores_from_mask(scr.scores, tar_mask)
        non_mask = scr.score_mask.multiply(key.non)
        non = self._extract_scores_from_mask(scr.scores, non_mask)
        if key.spoof is None:
            spoof = np.empty((0,), dtype=scr.scores.dtype)
        else:
            spoof_mask = scr.score_mask.multiply(key.spoof)
            spoof = self._extract_scores_from_mask(scr.scores, spoof_mask)
        return tar, non, spoof

    def get_valid_scores(
        self, ndx: Optional[Union[TrialNdx, TrialKey, SparseTrialKey]] = None
    ) -> np.ndarray:
        if ndx is None:
            scr = self
        else:
            scr = self.align_with_ndx(ndx)

        scores = self._extract_scores_from_mask(scr.scores, scr.score_mask)
        return scores

    def get_class_sim(
        self,
        key: SparseTrialKey,
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
        tar_mask = scr.score_mask.multiply(key.tar)
        non_mask = scr.score_mask.multiply(key.non)
        score_mask = tar_mask + non_mask  # still csr
        score_mask = score_mask.astype(np.float32)

        if model_classes is None or seg_classes is None:
            logging.info(
                "model/seg classes not provided, building from key.tar, it can take a while"
            )
            # Get row and column class labels from key.tar (dense helper handles that)
            model_classes, seg_classes = build_class_labels_from_boolean_matrix_sparse(
                key.tar
            )

        unique_model_classes = np.unique(model_classes)
        unique_seg_classes = np.unique(seg_classes)

        # Initialize the output similarity matrix
        sim_matrix = np.full(
            (len(unique_model_classes), len(unique_seg_classes)),
            np.nan,
            dtype=scr.scores.dtype,
        )

        # Loop over class pairs and compute masked mean
        for i, rc in enumerate(unique_model_classes):
            row_mask = model_classes == rc
            for j, cc in enumerate(unique_seg_classes):
                col_mask = seg_classes == cc

                # Get submatrices using slicing (convert masks to indices)
                row_idx = np.where(row_mask)[0]
                col_idx = np.where(col_mask)[0]
                if row_idx.size == 0 or col_idx.size == 0:
                    continue

                # Extract blocks
                score_block = scr.scores[row_idx[:, None], col_idx]
                mask_block = score_mask[row_idx[:, None], col_idx]

                # Compute masked average
                block_sum = score_block.multiply(mask_block).sum()
                count = mask_block.sum()

                if count > 0:
                    sim_matrix[i, j] = block_sum / count

        return sim_matrix, unique_model_classes, unique_seg_classes

    def set_valid_scores(
        self,
        scores: Union[np.ndarray, List[float]],
        ndx: Optional[Union[TrialNdx, TrialKey, SparseTrialKey]] = None,
    ) -> None:
        if ndx is not None:
            scr = self.align_with_ndx(ndx)
            self.model_set = scr.model_set
            self.seg_set = scr.seg_set
            self.scores = scr.scores
            self.score_mask = scr.score_mask

        self.scores[self.score_mask] = scores

    @classmethod
    def from_trial_scores(cls, scr: TrialScores) -> "SparseTrialScores":
        scores = scr.scores * scr.score_mask
        scores = sparse.csr_matrix(scores)
        score_mask = sparse.csr_matrix(scr.score_mask)
        scores.eliminate_zeros()
        score_mask.eliminate_zeros()
        scores.sort_indices()
        score_mask.sort_indices()
        return cls(scr.model_set, scr.seg_set, scores, score_mask)

    def to_trial_scores(self) -> TrialScores:
        scores = self.scores.toarray("C")
        score_mask = self.score_mask.toarray("C")
        # scores[~score_mask] = 0.0
        return TrialScores(self.model_set, self.seg_set, scores, score_mask)

    def set_missing_to_value(
        self,
        ndx: Union[TrialNdx, TrialKey, SparseTrialKey],
        val: float,
    ) -> "SparseTrialScores":
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
            if ndx.spoof is not None:
                mask = mask.maximum(ndx.spoof)
        elif isinstance(ndx, TrialKey):
            mask_dense = np.logical_or(ndx.tar, ndx.non)
            if ndx.spoof is not None:
                mask_dense = np.logical_or(mask_dense, ndx.spoof)
            mask = sparse.csr_matrix(mask_dense)
        else:
            raise ValueError(f"Unsupported ndx type: {type(ndx).__name__}")

        mask.eliminate_zeros()
        mask_coo = mask.tocoo()
        for r, c in zip(mask_coo.row, mask_coo.col):
            if not scr.score_mask[r, c]:
                scr.scores[r, c] = val

        scr.score_mask = mask
        return scr

    def __eq__(self, other: object) -> bool:
        """Equal operator"""
        if not isinstance(other, SparseTrialScores):
            return False
        self_scores = self.scores.tocsr()
        other_scores = other.scores.tocsr()
        self_mask = self.score_mask.tocsr()
        other_mask = other.score_mask.tocsr()
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and (self_scores.shape == other_scores.shape)
        eq = eq and (self_mask.shape == other_mask.shape)
        eq = eq and (self_scores.data.shape == other_scores.data.shape)
        eq = eq and (self_scores.indices.shape == other_scores.indices.shape)
        eq = eq and (self_scores.indptr.shape == other_scores.indptr.shape)
        eq = eq and (self_mask.data.shape == other_mask.data.shape)
        eq = eq and (self_mask.indices.shape == other_mask.indices.shape)
        eq = eq and (self_mask.indptr.shape == other_mask.indptr.shape)
        eq = eq and np.all(
            np.isclose(self_scores.data, other_scores.data, atol=1e-4, rtol=0.1)
        )
        eq = eq and np.all(self_scores.indices == other_scores.indices)
        eq = eq and np.all(self_scores.indptr == other_scores.indptr)
        eq = eq and np.all(self_mask.data == other_mask.data)
        eq = eq and np.all(self_mask.indices == other_mask.indices)
        eq = eq and np.all(self_mask.indptr == other_mask.indptr)
        return eq
