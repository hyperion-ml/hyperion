"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

# from .list_utils import *
from .list_utils import intersect, ismember, list2ndarray, sort, split_list
from .misc import PathLike

StrArrayLike = Union[np.ndarray, List[str]]


class TrialNdx:
    """Contains the trial index to run speaker recognition trials.

    BOSARIS-compatible trial index.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      trial_mask: Boolean matrix with the trials to execute to True (num_models x num_segments).

    Examples:
      >>> import numpy as np
      >>> from hyperion.utils.trial_ndx import TrialNdx
      >>> ndx = TrialNdx(
      ...     model_set=["m1", "m2"],
      ...     seg_set=["s1", "s2", "s3"],
      ...     trial_mask=np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
      ... )
      >>> ndx.num_models, ndx.num_tests
      (2, 3)
      >>> ndx_part = ndx.split(1, 2, 1, 1)
      >>> ndx_part.trial_mask.shape
      (1, 3)
    """

    def __init__(
        self,
        model_set: Optional[StrArrayLike] = None,
        seg_set: Optional[StrArrayLike] = None,
        trial_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.model_set = model_set
        self.seg_set = seg_set
        self.trial_mask = trial_mask
        if (model_set is not None) and (seg_set is not None):
            self.validate()

    @property
    def num_models(self) -> int:
        return len(self.model_set)

    @property
    def num_tests(self) -> int:
        return len(self.seg_set)

    def copy(self) -> "TrialNdx":
        """Makes a copy of the object"""
        return copy.deepcopy(self)

    def sort(self) -> None:
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        self.trial_mask = self.trial_mask[np.ix_(m_idx, s_idx)]

    def save(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves object to txt/h5 file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix
        if file_ext in [".h5", ".hdf5"]:
            self.save_h5(file_path)
        elif file_ext in [".txt", ""]:
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
            f.create_dataset("trial_mask", data=self.trial_mask.astype("uint8"))

    def save_txt(self, file_path: PathLike) -> None:
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        idx = (self.trial_mask.T == True).nonzero()
        with open(file_path, "w") as f:
            for item in zip(idx[0], idx[1]):
                f.write("%s %s\n" % (self.model_set[item[1]], self.seg_set[item[0]]))

    def save_table(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves object to a pandas table file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"modelid{sep}segmentid\n")
            I, J = self.trial_mask.nonzero()
            for i, j in zip(I, J):
                f.write(f"{self.model_set[i]}{sep}{self.seg_set[j]}\n")

    @classmethod
    def load(cls, file_path: PathLike, sep: Optional[str] = None) -> "TrialNdx":
        """Loads object from txt/h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialNdx object.
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
    def load_h5(cls, file_path: PathLike) -> "TrialNdx":
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
    def load_txt(cls, file_path: PathLike) -> "TrialNdx":
        """Loads object from txt file

        Args:
          file_path: File to read the list.

        Returns:
          TrialNdx object.
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
        trial_mask = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        for item in zip(model_idx, seg_idx):
            trial_mask[item[0], item[1]] = True
        return cls(model_set, seg_set, trial_mask)

    @classmethod
    def load_table(cls, file_path: PathLike, sep: Optional[str] = None) -> "TrialNdx":
        """Loads object from pandas table file

        Args:
          file_path: File to read the list.

        Returns:
          TrialNdx object.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        df = pd.read_csv(file_path, sep=sep, dtype={"modelid": str, "segmentid": str})
        models = df["modelid"].values
        segments = df["segmentid"].values
        model_set, _, model_idx = np.unique(
            models, return_index=True, return_inverse=True
        )
        seg_set, _, seg_idx = np.unique(
            segments, return_index=True, return_inverse=True
        )
        trial_mask = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        for i, j in zip(model_idx, seg_idx):
            trial_mask[i, j] = True

        return cls(model_set, seg_set, trial_mask)

    @classmethod
    def merge(cls, ndx_list: List["TrialNdx"]) -> "TrialNdx":
        """Merges several index objects.

        Args:
          ndx_list: List of TrialNdx objects.

        Returns:
          Merged TrialNdx object.
        """
        if len(ndx_list) == 0:
            raise ValueError("ndx_list must contain at least one TrialNdx")
        if len(ndx_list) == 1:
            return ndx_list[0].copy()

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
    def parse_eval_set(
        ndx: "TrialNdx",
        enroll: object,
        test: Optional[object] = None,
        eval_set: str = "enroll-test",
    ) -> Tuple["TrialNdx", object]:
        """Prepares the data structures required for evaluation.

        Args:
          ndx: TrialNdx object containing the trials for the main evaluation.
          enroll: Utt2Info where key are file_ids and second column are model names
          test: Utt2Info of where key are test segments names.
                Needed in the cases enroll-coh and coh-coh.
          eval_set: Type of evaluation
            enroll-test: main evaluation of enrollment vs test segments.
            enroll-coh: enrollment vs cohort segments.
            coh-test: cohort vs test segments.
            coh-coh: cohort vs cohort segments.

        Return:
          ndx: TrialNdx object
          enroll: SCPList
        """
        valid_eval_sets = {"enroll-test", "enroll-coh", "coh-test", "coh-coh"}
        if eval_set not in valid_eval_sets:
            raise ValueError(f"Unsupported eval_set='{eval_set}'")

        if eval_set in {"enroll-coh", "coh-coh"} and test is None:
            raise ValueError(f"test must be provided for eval_set='{eval_set}'")

        if eval_set == "enroll-test":
            enroll = enroll.filter_info(ndx.model_set)
        elif eval_set == "enroll-coh":
            ndx = TrialNdx(ndx.model_set, test.file_path)
            enroll = enroll.filter_info(ndx.model_set)
        elif eval_set == "coh-test":
            ndx = TrialNdx(enroll.key, ndx.seg_set)
        else:  # eval_set == "coh-coh"
            ndx = TrialNdx(enroll.key, test.file_path)
        return ndx, enroll

    def filter(
        self,
        model_set: StrArrayLike,
        seg_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "TrialNdx":
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
        trial_mask = self.trial_mask[np.ix_(mod_idx, seg_idx)]
        return TrialNdx(model_set, seg_set, trial_mask)

    def filter_by_model(
        self,
        model_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "TrialNdx":
        """Removes elements from TrialNdx object.

        Args:
          model_set: List of models to keep or remove.
          keep: If True, we keep the elements in model_set/seg_set,
                if False, we remove the elements in model_set/seg_set.

        Returns:
          Filtered TrialNdx object.
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
        trial_mask = self.trial_mask[mod_idx]
        return TrialNdx(model_set, self.seg_set, trial_mask)

    def split(
        self, model_idx: int, num_model_parts: int, seg_idx: int, num_seg_parts: int
    ) -> "TrialNdx":
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

    def validate(self) -> None:
        """Validates the attributes of the TrialNdx object."""
        self.model_set = list2ndarray(self.model_set)
        self.seg_set = list2ndarray(self.seg_set)

        if len(np.unique(self.model_set)) != len(self.model_set):
            raise ValueError("model_set must contain unique entries")
        if len(np.unique(self.seg_set)) != len(self.seg_set):
            raise ValueError("seg_set must contain unique entries")
        if self.trial_mask is None:
            self.trial_mask = np.ones(
                (len(self.model_set), len(self.seg_set)), dtype="bool"
            )
        else:
            expected_shape = (len(self.model_set), len(self.seg_set))
            if self.trial_mask.shape != expected_shape:
                raise ValueError(
                    f"trial_mask shape {self.trial_mask.shape} does not match {expected_shape}"
                )

    def apply_segmentation_to_test(self, segment_list: object) -> "TrialNdx":
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

    def __eq__(self, other: object) -> bool:
        """Equal operator"""
        if not isinstance(other, TrialNdx):
            return False
        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(self.trial_mask == other.trial_mask)
        return eq

    def __ne__(self, other: object) -> bool:
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other: object) -> int:
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1

    def test(ndx_file: PathLike = "core-core_det5_ndx.h5") -> None:

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
