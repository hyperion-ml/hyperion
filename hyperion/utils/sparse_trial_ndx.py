"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from .list_utils import intersect, ismember, list2ndarray, sort, split_list
from .misc import PathLike
from .trial_ndx import TrialNdx

StrArrayLike = Union[np.ndarray, List[str]]


class SparseTrialNdx(TrialNdx):
    """Contains sparse trial indices for speaker recognition trials.
        Bosaris compatible Ndx.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      trial_mask: Sparse boolean matrix with the trials to execute to True
        (num_models x num_segments).

    Examples:
      >>> import numpy as np
      >>> import scipy.sparse as sparse
      >>> from hyperion.utils.sparse_trial_ndx import SparseTrialNdx
      >>> ndx = SparseTrialNdx(
      ...     model_set=["m1", "m2"],
      ...     seg_set=["s1", "s2", "s3"],
      ...     trial_mask=sparse.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)),
      ... )
      >>> ndx.num_models, ndx.num_tests
      (2, 3)
      >>> ndx_part = ndx.split(1, 2, 1, 1)
      >>> ndx_part.trial_mask.shape
      (1, 3)
    """

    def __init__(
        self,
        model_set: Optional[StrArrayLike],
        seg_set: Optional[StrArrayLike],
        trial_mask: sparse.spmatrix,
    ) -> None:
        if trial_mask is None:
            raise ValueError("trial_mask cannot be None for SparseTrialNdx")
        super().__init__(model_set=model_set, seg_set=seg_set, trial_mask=trial_mask)

    @staticmethod
    def _full_trial_mask(num_models: int, num_tests: int) -> sparse.csr_matrix:
        """Creates an all-True sparse mask without allocating a dense matrix."""
        if num_models == 0 or num_tests == 0:
            return sparse.csr_matrix((num_models, num_tests), dtype="bool")
        nnz = num_models * num_tests
        data = np.ones(nnz, dtype="bool")
        indices = np.tile(np.arange(num_tests, dtype=np.int64), num_models)
        indptr = np.arange(0, nnz + 1, num_tests, dtype=np.int64)
        return sparse.csr_matrix((data, indices, indptr), shape=(num_models, num_tests))

    def sort(self) -> None:
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        trial_mask = self.trial_mask.tocsr()[m_idx][:, s_idx].tocsr()
        trial_mask.eliminate_zeros()
        trial_mask.sort_indices()
        self.trial_mask = trial_mask

    def save_h5(self, file_path: PathLike) -> None:
        raise NotImplementedError()

    def save_txt(self, file_path: PathLike) -> None:
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        trial_mask = self.trial_mask.tocsr()
        trial_mask.eliminate_zeros()
        trial_mask = trial_mask.tocoo()
        valid = np.asarray(trial_mask.data, dtype="bool")
        if np.all(valid):
            rows = trial_mask.row
            cols = trial_mask.col
        else:
            rows = trial_mask.row[valid]
            cols = trial_mask.col[valid]
        with open(file_path, "w") as f:
            for i, j in zip(rows, cols):
                f.write(f"{self.model_set[i]} {self.seg_set[j]}\n")

    def save_table(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves object to pandas table file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"modelid{sep}segmentid\n")
            trial_mask = self.trial_mask.tocsr()
            trial_mask.eliminate_zeros()
            trial_mask = trial_mask.tocoo()
            valid = np.asarray(trial_mask.data, dtype="bool")
            if np.all(valid):
                rows = trial_mask.row
                cols = trial_mask.col
            else:
                rows = trial_mask.row[valid]
                cols = trial_mask.col[valid]
            for i, j in zip(rows, cols):
                f.write(f"{self.model_set[i]}{sep}{self.seg_set[j]}\n")

    @classmethod
    def load_h5(cls, file_path: PathLike) -> "SparseTrialNdx":
        raise NotImplementedError()

    @classmethod
    def load_txt(cls, file_path: PathLike) -> "SparseTrialNdx":
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          SparseTrialNdx object.
        """
        rows = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) == 0:
                    continue
                if len(parts) < 2:
                    raise ValueError(
                        f"Malformed line {line_num} in ndx file: expected at least 2 columns"
                    )
                rows.append((parts[0], parts[1]))

        models = [r[0] for r in rows]
        segments = [r[1] for r in rows]
        model_set, _, model_idx = np.unique(
            models, return_index=True, return_inverse=True
        )
        seg_set, _, seg_idx = np.unique(
            segments, return_index=True, return_inverse=True
        )
        trial_mask = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        for i, j in zip(model_idx, seg_idx):
            trial_mask[i, j] = True
        return cls(model_set, seg_set, trial_mask.tocsr())

    @classmethod
    def load_table(
        cls, file_path: PathLike, sep: Optional[str] = None
    ) -> "SparseTrialNdx":
        """Loads object from pandas table file.

        Args:
          file_path: File to read the list.

        Returns:
          SparseTrialNdx object.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        df = pd.read_csv(file_path, sep=sep, dtype={"modelid": str, "segmentid": str})
        models = df["modelid"].values
        segments = df["segmentid"].values
        model_set, model_idx = np.unique(models, return_inverse=True)
        seg_set, seg_idx = np.unique(segments, return_inverse=True)
        trial_mask = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        for i, j in zip(model_idx, seg_idx):
            trial_mask[i, j] = True
        return cls(model_set, seg_set, trial_mask.tocsr())

    @classmethod
    def merge(cls, ndx_list: List["SparseTrialNdx"]) -> "SparseTrialNdx":
        """Merges several index objects.

        Args:
          ndx_list: List of SparseTrialNdx objects.

        Returns:
          Merged SparseTrialNdx object.
        """
        if len(ndx_list) == 0:
            raise ValueError("ndx_list must contain at least one SparseTrialNdx")
        if len(ndx_list) == 1:
            return ndx_list[0].copy()

        model_set = ndx_list[0].model_set
        seg_set = ndx_list[0].seg_set
        trial_mask = ndx_list[0].trial_mask.tocsr()
        trial_mask.eliminate_zeros()
        trial_mask.sort_indices()

        for i in range(1, len(ndx_list)):
            ndx_i = ndx_list[i]
            new_model_set = np.union1d(model_set, ndx_i.model_set)
            new_seg_set = np.union1d(seg_set, ndx_i.seg_set)
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
            trial_mask_1 = trial_mask.tocoo()
            trial_mask_1 = sparse.coo_matrix(
                (
                    trial_mask_1.data,
                    (
                        model_map_1[trial_mask_1.row],
                        seg_map_1[trial_mask_1.col],
                    ),
                ),
                shape=shape,
                dtype="bool",
            ).tocsr()
            trial_mask_1.eliminate_zeros()
            trial_mask_1.sort_indices()

            _, mi_a, mi_b = intersect(
                new_model_set, ndx_i.model_set, assume_unique=True, return_index=True
            )
            _, si_a, si_b = intersect(
                new_seg_set, ndx_i.seg_set, assume_unique=True, return_index=True
            )
            model_map_2 = np.empty(len(ndx_i.model_set), dtype="int64")
            seg_map_2 = np.empty(len(ndx_i.seg_set), dtype="int64")
            model_map_2[mi_b] = mi_a
            seg_map_2[si_b] = si_a

            trial_mask_i = ndx_i.trial_mask.tocsr()
            trial_mask_i.eliminate_zeros()
            trial_mask_i.sort_indices()
            trial_mask_2 = trial_mask_i.tocoo()
            trial_mask_2 = sparse.coo_matrix(
                (
                    trial_mask_2.data,
                    (
                        model_map_2[trial_mask_2.row],
                        seg_map_2[trial_mask_2.col],
                    ),
                ),
                shape=shape,
                dtype="bool",
            ).tocsr()
            trial_mask_2.eliminate_zeros()
            trial_mask_2.sort_indices()

            model_set = new_model_set
            seg_set = new_seg_set
            trial_mask = trial_mask_1.maximum(trial_mask_2).tocsr()
            trial_mask.eliminate_zeros()
            trial_mask.sort_indices()

        return cls(model_set, seg_set, trial_mask)

    @staticmethod
    def parse_eval_set(
        ndx: "SparseTrialNdx",
        enroll: object,
        test: Optional[object] = None,
        eval_set: str = "enroll-test",
    ) -> Tuple["SparseTrialNdx", object]:
        """Prepares sparse data structures required for evaluation."""
        valid_eval_sets = {"enroll-test", "enroll-coh", "coh-test", "coh-coh"}
        if eval_set not in valid_eval_sets:
            raise ValueError(f"Unsupported eval_set='{eval_set}'")

        if eval_set in {"enroll-coh", "coh-coh"} and test is None:
            raise ValueError(f"test must be provided for eval_set='{eval_set}'")

        if eval_set == "enroll-test":
            enroll = enroll.filter_info(ndx.model_set)
        elif eval_set == "enroll-coh":
            model_set = list2ndarray(ndx.model_set)
            seg_set = list2ndarray(test.file_path)
            trial_mask = SparseTrialNdx._full_trial_mask(len(model_set), len(seg_set))
            ndx = SparseTrialNdx(model_set, seg_set, trial_mask)
            enroll = enroll.filter_info(ndx.model_set)
        elif eval_set == "coh-test":
            model_set = list2ndarray(enroll.key)
            seg_set = list2ndarray(ndx.seg_set)
            trial_mask = SparseTrialNdx._full_trial_mask(len(model_set), len(seg_set))
            ndx = SparseTrialNdx(model_set, seg_set, trial_mask)
        else:  # eval_set == "coh-coh"
            model_set = list2ndarray(enroll.key)
            seg_set = list2ndarray(test.file_path)
            trial_mask = SparseTrialNdx._full_trial_mask(len(model_set), len(seg_set))
            ndx = SparseTrialNdx(model_set, seg_set, trial_mask)
        return ndx, enroll

    def filter(
        self,
        model_set: StrArrayLike,
        seg_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "SparseTrialNdx":
        """Removes elements from SparseTrialNdx object.

        Args:
          model_set: List of models to keep or remove.
          seg_set: List of test segments to keep or remove.
          keep: If True, keeps elements in model_set/seg_set.
          raise_missing: Raises error if requested models or segments are missing.

        Returns:
          Filtered SparseTrialNdx object.
        """
        if not keep:
            model_set = np.setdiff1d(self.model_set, model_set)
            seg_set = np.setdiff1d(self.seg_set, seg_set)

        f, mod_idx = ismember(model_set, self.model_set)
        if raise_missing:
            if not np.all(f):
                missing_models = np.asarray(model_set)[~f]
                raise ValueError(f"models not found: {missing_models.tolist()}")
        else:
            mod_idx = mod_idx[f]

        f, seg_idx = ismember(seg_set, self.seg_set)
        if raise_missing:
            if not np.all(f):
                missing_segs = np.asarray(seg_set)[~f]
                raise ValueError(f"segments not found: {missing_segs.tolist()}")
        else:
            seg_idx = seg_idx[f]

        model_set = self.model_set[mod_idx]
        seg_set = self.seg_set[seg_idx]
        trial_mask = self.trial_mask.tocsr()[mod_idx][:, seg_idx].tocsr()
        trial_mask.eliminate_zeros()
        trial_mask.sort_indices()
        return SparseTrialNdx(model_set, seg_set, trial_mask)

    def filter_by_model(
        self,
        model_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "SparseTrialNdx":
        """Removes model entries from SparseTrialNdx object.

        Args:
          model_set: List of models to keep or remove.
          keep: If True, keeps elements in model_set.
          raise_missing: Raises error if requested models are missing.

        Returns:
          Filtered SparseTrialNdx object.
        """
        if not keep:
            model_set = np.setdiff1d(self.model_set, model_set)

        f, mod_idx = ismember(model_set, self.model_set)
        if raise_missing:
            if not np.all(f):
                missing_models = np.asarray(model_set)[~f]
                raise ValueError(f"models not found: {missing_models.tolist()}")
        else:
            mod_idx = mod_idx[f]

        model_set = self.model_set[mod_idx]
        trial_mask = self.trial_mask.tocsr()[mod_idx].tocsr()
        trial_mask.eliminate_zeros()
        trial_mask.sort_indices()
        return SparseTrialNdx(model_set, self.seg_set, trial_mask)

    def split(
        self, model_idx: int, num_model_parts: int, seg_idx: int, num_seg_parts: int
    ) -> "SparseTrialNdx":
        """Splits the object and returns one subpart."""
        model_set, model_idx1 = split_list(self.model_set, model_idx, num_model_parts)
        seg_set, seg_idx1 = split_list(self.seg_set, seg_idx, num_seg_parts)
        trial_mask = self.trial_mask.tocsr()[model_idx1][:, seg_idx1].tocsr()
        trial_mask.eliminate_zeros()
        trial_mask.sort_indices()
        return SparseTrialNdx(model_set, seg_set, trial_mask)

    def validate(self) -> None:
        """Validates the attributes of the SparseTrialNdx object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        if len(np.unique(self.model_set)) != len(self.model_set):
            raise ValueError("model_set must contain unique entries")
        if len(np.unique(self.seg_set)) != len(self.seg_set):
            raise ValueError("seg_set must contain unique entries")

        shape = (len(self.model_set), len(self.seg_set))
        if self.trial_mask is None:
            raise ValueError("trial_mask cannot be None for SparseTrialNdx")

        self.trial_mask = self.trial_mask.tocsr().astype("bool")
        if self.trial_mask.shape != shape:
            raise ValueError(
                f"trial_mask shape {self.trial_mask.shape} does not match {shape}"
            )

        self.trial_mask.eliminate_zeros()
        self.trial_mask.sort_indices()

    def apply_segmentation_to_test(self, segment_list: object) -> "SparseTrialNdx":
        """Splits test segments into multiple sub-segments.

        Args:
          segment_list: ExtSegmentList object with mapping of file_id to
            ext_segment_id.

        Returns:
          New SparseTrialNdx with segment_ids in test instead of file_id.
        """
        new_segset = []
        new_mask_parts = []
        trial_mask = self.trial_mask.tocsc()
        for i in range(self.num_tests):
            file_id = self.seg_set[i]
            segment_ids = segment_list.ext_segment_ids_from_file(file_id)
            if len(segment_ids) == 0:
                continue
            new_segset.append(segment_ids)
            col_mask = trial_mask[:, i].tocsr()
            rep_mask = sparse.hstack([col_mask] * len(segment_ids), format="csr")
            new_mask_parts.append(rep_mask)

        if len(new_segset) == 0:
            seg_dtype = self.seg_set.dtype if hasattr(self.seg_set, "dtype") else str
            new_segset_arr = np.asarray([], dtype=seg_dtype)
            new_mask = sparse.csr_matrix((self.num_models, 0), dtype="bool")
        else:
            new_segset_arr = np.concatenate(tuple(new_segset))
            new_mask = sparse.hstack(new_mask_parts, format="csr")
            new_mask.eliminate_zeros()
            new_mask.sort_indices()

        return SparseTrialNdx(self.model_set, new_segset_arr, new_mask)

    @classmethod
    def from_trial_ndx(cls, ndx: TrialNdx) -> "SparseTrialNdx":
        """Builds a SparseTrialNdx from a dense TrialNdx."""
        trial_mask = sparse.csr_matrix(ndx.trial_mask)
        trial_mask.eliminate_zeros()
        trial_mask.sort_indices()
        return cls(ndx.model_set, ndx.seg_set, trial_mask)

    def to_trial_ndx(self) -> TrialNdx:
        """Converts SparseTrialNdx to dense TrialNdx."""
        trial_mask = self.trial_mask.toarray()
        return TrialNdx(self.model_set, self.seg_set, trial_mask)

    def __eq__(self, other: object) -> bool:
        """Equal operator."""
        if not isinstance(other, SparseTrialNdx):
            return False
        self_mask = self.trial_mask.tocsr()
        other_mask = other.trial_mask.tocsr()
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and (self_mask.shape == other_mask.shape)
        eq = eq and (self_mask.data.shape == other_mask.data.shape)
        eq = eq and (self_mask.indices.shape == other_mask.indices.shape)
        eq = eq and (self_mask.indptr.shape == other_mask.indptr.shape)
        eq = eq and np.all(self_mask.data == other_mask.data)
        eq = eq and np.all(self_mask.indices == other_mask.indices)
        eq = eq and np.all(self_mask.indptr == other_mask.indptr)
        return eq

    def __ne__(self, other: object) -> bool:
        """Non-equal operator."""
        return not self.__eq__(other)

    def __cmp__(self, other: object) -> int:
        """Comparison operator."""
        if self.__eq__(other):
            return 0
        return 1
