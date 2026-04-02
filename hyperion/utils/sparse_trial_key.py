"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from .list_utils import *
from .misc import PathLike
from .trial_key import TrialKey
from .trial_ndx import TrialNdx

StrArrayLike = Union[np.ndarray, List[str]]


class SparseTrialKey(TrialKey):
    """Contains the trial key for speaker recognition trials.
        Bosaris compatible Key.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      tar: Boolean matrix with target trials to True (num_models x num_segments).
      non: Boolean matrix with non-target trials to True (num_models x num_segments).
      spoof: Boolean matrix with spoof trials to True (num_models x num_segments).
      model_cond: Conditions related to the model.
      seg_cond: Conditions related to the test segment.
      trial_cond: Conditions related to the combination of model and test segment.
      model_cond_name: String list with the names of the model conditions.
      seg_cond_name: String list with the names of the segment conditions.
      trial_cond_name: String list with the names of the trial conditions.

    Examples:
      >>> import numpy as np
      >>> import scipy.sparse as sparse
      >>> from hyperion.utils.sparse_trial_key import SparseTrialKey
      >>> tar = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=bool))
      >>> non = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool))
      >>> key = SparseTrialKey(model_set=["m1", "m2"], seg_set=["s1", "s2"], tar=tar, non=non)
      >>> ndx = key.to_ndx()
      >>> ndx.trial_mask.shape
      (2, 2)
      >>> from hyperion.utils.trial_key import TrialKey
      >>> dense_key = TrialKey(
      ...     model_set=["m1", "m2"],
      ...     seg_set=["s1", "s2"],
      ...     tar=np.array([[1, 0], [0, 1]], dtype=bool),
      ...     non=np.array([[0, 1], [1, 0]], dtype=bool),
      ... )
      >>> sparse_key = SparseTrialKey.from_trial_key(dense_key)
      >>> sparse_key.tar.nnz
      2
    """

    def __init__(
        self,
        model_set: Optional[StrArrayLike] = None,
        seg_set: Optional[StrArrayLike] = None,
        tar: Optional[sparse.spmatrix] = None,
        non: Optional[sparse.spmatrix] = None,
        spoof: Optional[sparse.spmatrix] = None,
        model_cond: Optional[np.ndarray] = None,
        seg_cond: Optional[np.ndarray] = None,
        trial_cond: Optional[np.ndarray] = None,
        model_cond_name: Optional[StrArrayLike] = None,
        seg_cond_name: Optional[StrArrayLike] = None,
        trial_cond_name: Optional[StrArrayLike] = None,
    ) -> None:

        super().__init__(
            model_set=model_set,
            seg_set=seg_set,
            tar=tar,
            non=non,
            spoof=spoof,
            model_cond=model_cond,
            seg_cond=seg_cond,
            trial_cond=trial_cond,
            model_cond_name=model_cond_name,
            seg_cond_name=seg_cond_name,
            trial_cond_name=trial_cond_name,
        )

    def sort(self) -> None:
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        tar = self.tar.tocsr()[m_idx][:, s_idx].tocsr()
        non = self.non.tocsr()[m_idx][:, s_idx].tocsr()
        spoof = None
        if self.spoof is not None:
            spoof = self.spoof.tocsr()[m_idx][:, s_idx].tocsr()
        tar.eliminate_zeros()
        non.eliminate_zeros()
        if spoof is not None:
            spoof.eliminate_zeros()
        tar.sort_indices()
        non.sort_indices()
        if spoof is not None:
            spoof.sort_indices()
        self.tar = tar
        self.non = non
        self.spoof = spoof
        if self.model_cond is not None:
            self.model_cond = self.model_cond[:, m_idx]
        if self.seg_cond is not None:
            self.seg_cond = self.seg_cond[:, s_idx]
        if self.trial_cond is not None:
            self.trial_cond = self.trial_cond[:, m_idx][:, :, s_idx]

    def save_h5(self, file_path: PathLike) -> None:
        raise NotImplementedError()

    def save_txt(self, file_path: PathLike) -> None:
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        with open(file_path, "w") as f:
            self.tar.eliminate_zeros()
            self.non.eliminate_zeros()
            if self.spoof is not None:
                self.spoof.eliminate_zeros()
            tar = self.tar.tocoo()
            for r, c in zip(tar.row, tar.col):
                f.write("%s %s target\n" % (self.model_set[r], self.seg_set[c]))

            non = self.non.tocoo()
            for r, c in zip(non.row, non.col):
                f.write("%s %s nontarget\n" % (self.model_set[r], self.seg_set[c]))

            if self.spoof is not None:
                spoof = self.spoof.tocoo()
                for r, c in zip(spoof.row, spoof.col):
                    f.write("%s %s spoof\n" % (self.model_set[r], self.seg_set[c]))

    def save_table(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"modelid{sep}segmentid{sep}targettype\n")
            self.tar.eliminate_zeros()
            self.non.eliminate_zeros()
            if self.spoof is not None:
                self.spoof.eliminate_zeros()
                non = self.non.maximum(self.spoof)
            else:
                non = self.non
            mask = self.tar.maximum(non).tocoo()
            for r, c in zip(mask.row, mask.col):
                target_type = (
                    "target"
                    if self.tar[r, c]
                    else ("nontarget" if self.non[r, c] else "spoof")
                )
                f.write(
                    f"{self.model_set[r]}{sep}{self.seg_set[c]}{sep}{target_type}\n"
                )

    @classmethod
    def load_h5(cls, file_path: PathLike) -> "SparseTrialKey":
        raise NotImplementedError()

    @classmethod
    def load_txt(cls, file_path: PathLike) -> "SparseTrialKey":
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          TrialKey object.
        """
        fields = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) == 0:
                    continue
                if len(parts) < 3:
                    raise ValueError(
                        f"Malformed line {line_num} in key file: expected at least 3 columns"
                    )
                fields.append(parts[:3])

        models = [i[0] for i in fields]
        segments = [i[1] for i in fields]
        labels = [i[2] for i in fields]
        valid_labels = {"target", "nontarget", "spoof"}
        invalid_labels = sorted({l for l in labels if l not in valid_labels})
        if invalid_labels:
            raise ValueError(
                f"Invalid target labels in key file: {invalid_labels}. "
                "Expected one of ['target', 'nontarget', 'spoof']"
            )
        is_tar = [l == "target" for l in labels]
        is_non = [l == "nontarget" for l in labels]
        is_spoof = [l == "spoof" for l in labels]
        model_set, _, model_idx = np.unique(
            models, return_index=True, return_inverse=True
        )
        seg_set, _, seg_idx = np.unique(
            segments, return_index=True, return_inverse=True
        )
        tar = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        non = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        if np.any(is_spoof):
            spoof = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        else:
            spoof = None
        for item in zip(model_idx, seg_idx, is_tar, is_non, is_spoof):
            if item[2]:
                tar[item[0], item[1]] = True
            elif item[3]:
                non[item[0], item[1]] = True
            elif item[4]:
                spoof[item[0], item[1]] = True
            else:
                raise ValueError("Invalid target label encountered while parsing key")
        spoof = spoof.tocsr() if spoof is not None else None
        return cls(model_set, seg_set, tar.tocsr(), non.tocsr(), spoof=spoof)

    @classmethod
    def load_table(
        cls, file_path: PathLike, sep: Optional[str] = None
    ) -> "SparseTrialKey":
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          SparseTrialKey object.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        df = pd.read_csv(file_path, sep=sep, dtype={"modelid": str, "segmentid": str})
        models = df["modelid"].values
        segments = df["segmentid"].values
        labels = df["targettype"].astype(str).values
        valid_labels = {"target", "nontarget", "spoof"}
        invalid_labels = sorted(set(labels) - valid_labels)
        if invalid_labels:
            raise ValueError(
                f"Invalid target labels in key table: {invalid_labels}. "
                "Expected one of ['target', 'nontarget', 'spoof']"
            )
        is_tar = labels == "target"
        is_non = labels == "nontarget"
        is_spoof = labels == "spoof"
        model_set, model_idx = np.unique(models, return_inverse=True)
        seg_set, seg_idx = np.unique(segments, return_inverse=True)
        tar = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        non = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        if np.any(is_spoof):
            spoof = sparse.lil_matrix((len(model_set), len(seg_set)), dtype="bool")
        else:
            spoof = None
        for i, j, target_type, non_type, spoof_type in zip(
            model_idx, seg_idx, is_tar, is_non, is_spoof
        ):
            if target_type:
                tar[i, j] = True
            elif non_type:
                non[i, j] = True
            elif spoof_type:
                spoof[i, j] = True
            else:
                raise ValueError(
                    "Invalid target label encountered while parsing key table"
                )
        spoof = spoof.tocsr() if spoof is not None else None
        return cls(model_set, seg_set, tar.tocsr(), non.tocsr(), spoof=spoof)

    @classmethod
    def merge(cls, key_list: List["SparseTrialKey"]) -> "SparseTrialKey":
        raise NotImplementedError()

    def filter(
        self,
        model_set: StrArrayLike,
        seg_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "SparseTrialKey":
        """Removes elements from SparseTrialKey object."""

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
        ix = np.ix_(mod_idx, seg_idx)
        tar = self.tar[ix]
        non = self.non[ix]
        if self.spoof is not None:
            spoof = self.spoof[ix]
        else:
            spoof = None

        model_cond = None
        seg_cond = None
        trial_cond = None
        if self.model_cond is not None:
            model_cond = self.model_cond[:, mod_idx]
        if self.seg_cond is not None:
            seg_cond = self.seg_cond[:, seg_idx]
        if self.trial_cond is not None:
            trial_cond = self.trial_cond[:, mod_idx][:, :, seg_idx]

        return SparseTrialKey(
            model_set,
            seg_set,
            tar,
            non,
            spoof,
            model_cond,
            seg_cond,
            trial_cond,
            self.model_cond_name,
            self.seg_cond_name,
            self.trial_cond_name,
        )

    def filter_by_model(
        self,
        model_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "SparseTrialKey":
        """Removes model entries from SparseTrialKey object."""

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
        tar = self.tar[mod_idx]
        non = self.non[mod_idx]
        if self.spoof is not None:
            spoof = self.spoof[mod_idx]
        else:
            spoof = None

        model_cond = None
        seg_cond = None
        trial_cond = None
        if self.model_cond is not None:
            model_cond = self.model_cond[:, mod_idx]
        if self.seg_cond is not None:
            seg_cond = self.seg_cond
        if self.trial_cond is not None:
            trial_cond = self.trial_cond[:, mod_idx]

        return SparseTrialKey(
            model_set,
            self.seg_set,
            tar,
            non,
            spoof,
            model_cond,
            seg_cond,
            trial_cond,
            self.model_cond_name,
            self.seg_cond_name,
            self.trial_cond_name,
        )

    def split(
        self, model_idx: int, num_model_parts: int, seg_idx: int, num_seg_parts: int
    ) -> "SparseTrialKey":
        """Splits the SparseTrialKey and returns one subpart."""

        model_set, model_idx1 = split_list(self.model_set, model_idx, num_model_parts)
        seg_set, seg_idx1 = split_list(self.seg_set, seg_idx, num_seg_parts)
        ix = np.ix_(model_idx1, seg_idx1)
        tar = self.tar[ix]
        non = self.non[ix]
        if self.spoof is not None:
            spoof = self.spoof[ix]
        else:
            spoof = None

        model_cond = None
        seg_cond = None
        trial_cond = None
        if self.model_cond is not None:
            model_cond = self.model_cond[:, model_idx1]
        if self.seg_cond is not None:
            seg_cond = self.seg_cond[:, seg_idx1]
        if self.trial_cond is not None:
            trial_cond = self.trial_cond[:, model_idx1][:, :, seg_idx1]

        return SparseTrialKey(
            model_set,
            seg_set,
            tar,
            non,
            spoof,
            model_cond,
            seg_cond,
            trial_cond,
            self.model_cond_name,
            self.seg_cond_name,
            self.trial_cond_name,
        )

    def to_ndx(self) -> TrialNdx:
        """Converts TrialKey object into TrialNdx object.

        Returns:
          TrialNdx object.
        """
        mask = self.tar.maximum(self.non)
        if self.spoof is not None:
            mask = mask.maximum(self.spoof)
        mask = mask.toarray()
        return TrialNdx(self.model_set, self.seg_set, mask)

    def validate(self) -> None:
        """Validates the attributes of the TrialKey object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        shape = (len(self.model_set), len(self.seg_set))
        if len(np.unique(self.model_set)) != shape[0]:
            raise ValueError("model_set must contain unique entries")
        if len(np.unique(self.seg_set)) != shape[1]:
            raise ValueError("seg_set must contain unique entries")

        if (self.tar is None) or (self.non is None):
            self.tar = sparse.csr_matrix(shape, dtype="bool")
            self.non = sparse.csr_matrix(shape, dtype="bool")
        else:
            if self.tar.shape != shape:
                raise ValueError(f"tar shape {self.tar.shape} does not match {shape}")
            if self.non.shape != shape:
                raise ValueError(f"non shape {self.non.shape} does not match {shape}")
        if self.spoof is not None:
            if self.spoof.shape != shape:
                raise ValueError(
                    f"spoof shape {self.spoof.shape} does not match {shape}"
                )

        self.tar = self.tar.tocsr()
        self.non = self.non.tocsr()
        if self.spoof is not None:
            self.spoof = self.spoof.tocsr()
        self.tar.eliminate_zeros()
        self.non.eliminate_zeros()
        if self.spoof is not None:
            self.spoof.eliminate_zeros()
        self.tar.sort_indices()
        self.non.sort_indices()
        if self.spoof is not None:
            self.spoof.sort_indices()
        if self.tar.multiply(self.non).nnz > 0:
            raise ValueError("tar and non overlap")
        if self.spoof is not None:
            if self.tar.multiply(self.spoof).nnz > 0:
                raise ValueError("tar and spoof overlap")
            if self.non.multiply(self.spoof).nnz > 0:
                raise ValueError("non and spoof overlap")

        if self.model_cond is not None:
            if self.model_cond.shape[1] != shape[0]:
                raise ValueError(
                    f"model_cond second dimension {self.model_cond.shape[1]} "
                    f"does not match num_models {shape[0]}"
                )
        if self.seg_cond is not None:
            if self.seg_cond.shape[1] != shape[1]:
                raise ValueError(
                    f"seg_cond second dimension {self.seg_cond.shape[1]} "
                    f"does not match num_segments {shape[1]}"
                )
        if self.trial_cond is not None:
            if self.trial_cond.shape[1:] != shape:
                raise ValueError(
                    f"trial_cond shape {self.trial_cond.shape[1:]} "
                    f"does not match {(shape[0], shape[1])}"
                )

        if self.model_cond_name is not None:
            self.model_cond_name = list2ndarray(self.model_cond_name)
            if self.model_cond is None:
                raise ValueError("model_cond_name is set but model_cond is None")
            if len(self.model_cond_name) != self.model_cond.shape[0]:
                raise ValueError(
                    "model_cond_name length must match number of model conditions"
                )
        if self.seg_cond_name is not None:
            self.seg_cond_name = list2ndarray(self.seg_cond_name)
            if self.seg_cond is None:
                raise ValueError("seg_cond_name is set but seg_cond is None")
            if len(self.seg_cond_name) != self.seg_cond.shape[0]:
                raise ValueError(
                    "seg_cond_name length must match number of segment conditions"
                )
        if self.trial_cond_name is not None:
            self.trial_cond_name = list2ndarray(self.trial_cond_name)
            if self.trial_cond is None:
                raise ValueError("trial_cond_name is set but trial_cond is None")
            if len(self.trial_cond_name) != self.trial_cond.shape[0]:
                raise ValueError(
                    "trial_cond_name length must match number of trial conditions"
                )

    @classmethod
    def from_trial_key(cls, key: TrialKey) -> "SparseTrialKey":
        tar = sparse.csr_matrix(key.tar)
        non = sparse.csr_matrix(key.non)
        spoof = None
        if key.spoof is not None:
            spoof = sparse.csr_matrix(key.spoof)
        tar.eliminate_zeros()
        non.eliminate_zeros()
        if spoof is not None:
            spoof.eliminate_zeros()
        tar.sort_indices()
        non.sort_indices()
        if spoof is not None:
            spoof.sort_indices()
        return cls(
            key.model_set,
            key.seg_set,
            tar,
            non,
            spoof,
            key.model_cond,
            key.seg_cond,
            key.trial_cond,
            key.model_cond_name,
            key.seg_cond_name,
            key.trial_cond_name,
        )

    def __eq__(self, other: object) -> bool:
        """Equal operator"""
        if not isinstance(other, SparseTrialKey):
            return False

        self_tar = self.tar.tocsr()
        self_non = self.non.tocsr()
        other_tar = other.tar.tocsr()
        other_non = other.non.tocsr()
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and (self_tar.shape == other_tar.shape)
        eq = eq and (self_non.shape == other_non.shape)
        eq = eq and (self_tar.data.shape == other_tar.data.shape)
        eq = eq and (self_non.data.shape == other_non.data.shape)
        eq = eq and (self_tar.indices.shape == other_tar.indices.shape)
        eq = eq and (self_non.indices.shape == other_non.indices.shape)
        eq = eq and (self_tar.indptr.shape == other_tar.indptr.shape)
        eq = eq and (self_non.indptr.shape == other_non.indptr.shape)
        eq = eq and np.all(self_tar.data == other_tar.data)
        eq = eq and np.all(self_non.data == other_non.data)
        eq = eq and np.all(self_tar.indices == other_tar.indices)
        eq = eq and np.all(self_non.indices == other_non.indices)
        eq = eq and np.all(self_tar.indptr == other_tar.indptr)
        eq = eq and np.all(self_non.indptr == other_non.indptr)
        eq = eq and ((self.spoof is None) == (other.spoof is None))
        if self.spoof is not None and other.spoof is not None:
            self_spoof = self.spoof.tocsr()
            other_spoof = other.spoof.tocsr()
            eq = eq and (self_spoof.shape == other_spoof.shape)
            eq = eq and (self_spoof.data.shape == other_spoof.data.shape)
            eq = eq and (self_spoof.indices.shape == other_spoof.indices.shape)
            eq = eq and (self_spoof.indptr.shape == other_spoof.indptr.shape)
            eq = eq and np.all(self_spoof.data == other_spoof.data)
            eq = eq and np.all(self_spoof.indices == other_spoof.indices)
            eq = eq and np.all(self_spoof.indptr == other_spoof.indptr)

        eq = eq and ((self.model_cond is None) == (other.model_cond is None))
        eq = eq and ((self.seg_cond is None) == (other.seg_cond is None))
        eq = eq and ((self.trial_cond is None) == (other.trial_cond is None))

        if self.model_cond is not None:
            eq = eq and (self.model_cond.shape == other.model_cond.shape)
            eq = eq and np.all(self.model_cond == other.model_cond)
        if self.seg_cond is not None:
            eq = eq and (self.seg_cond.shape == other.seg_cond.shape)
            eq = eq and np.all(self.seg_cond == other.seg_cond)
        if self.trial_cond is not None:
            eq = eq and (self.trial_cond.shape == other.trial_cond.shape)
            eq = eq and np.all(self.trial_cond == other.trial_cond)

        eq = eq and ((self.model_cond_name is None) == (other.model_cond_name is None))
        eq = eq and ((self.seg_cond_name is None) == (other.seg_cond_name is None))
        eq = eq and ((self.trial_cond_name is None) == (other.trial_cond_name is None))

        if self.model_cond_name is not None:
            eq = eq and (self.model_cond_name.shape == other.model_cond_name.shape)
            eq = eq and np.all(self.model_cond_name == other.model_cond_name)
        if self.seg_cond_name is not None:
            eq = eq and (self.seg_cond_name.shape == other.seg_cond_name.shape)
            eq = eq and np.all(self.seg_cond_name == other.seg_cond_name)
        if self.trial_cond_name is not None:
            eq = eq and (self.trial_cond_name.shape == other.trial_cond_name.shape)
            eq = eq and np.all(self.trial_cond_name == other.trial_cond_name)

        return eq
