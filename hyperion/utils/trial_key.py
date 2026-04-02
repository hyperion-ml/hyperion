"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import os.path as path
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np
import pandas as pd

# from .list_utils import *
from .list_utils import intersect, ismember, list2ndarray, sort, split_list
from .misc import PathLike
from .trial_ndx import TrialNdx

StrArrayLike = Union[np.ndarray, List[str]]


class TrialKey:
    """Contains the trial key for speaker recognition trials.
        Bosaris compatible Key.

    Attributes:
      model_set: List of model names.
      seg_set: List of test segment names.
      tar: Boolean matrix with target trials to True (num_models x num_segments).
      non: Boolean matrix with non-target trials to True (num_models x num_segments).
      spoof: Boolean matrix with spoof trials to True (num_models x num_segments)
      model_cond: Conditions related to the model.
      seg_cond: Conditions related to the test segment.
      trial_cond: Conditions related to the combination of model and test segment.
      model_cond_name: String list with the names of the model conditions.
      seg_cond_name: String list with the names of the segment conditions.
      trial_cond_name: String list with the names of the trial conditions.

    Examples:
      >>> import numpy as np
      >>> from hyperion.utils.trial_key import TrialKey
      >>> key = TrialKey(
      ...     model_set=["m1", "m2"],
      ...     seg_set=["s1", "s2"],
      ...     tar=np.array([[1, 0], [0, 1]], dtype=bool),
      ...     non=np.array([[0, 1], [1, 0]], dtype=bool),
      ... )
      >>> ndx = key.to_ndx()
      >>> ndx.trial_mask.shape
      (2, 2)
      >>> key_small = key.filter(["m1"], ["s1", "s2"], keep=True)
      >>> key_small.tar.shape
      (1, 2)
    """

    def __init__(
        self,
        model_set: Optional[StrArrayLike] = None,
        seg_set: Optional[StrArrayLike] = None,
        tar: Optional[np.ndarray] = None,
        non: Optional[np.ndarray] = None,
        spoof: Optional[np.ndarray] = None,
        model_cond: Optional[np.ndarray] = None,
        seg_cond: Optional[np.ndarray] = None,
        trial_cond: Optional[np.ndarray] = None,
        model_cond_name: Optional[StrArrayLike] = None,
        seg_cond_name: Optional[StrArrayLike] = None,
        trial_cond_name: Optional[StrArrayLike] = None,
    ) -> None:
        self.model_set = model_set
        self.seg_set = seg_set
        self.tar = tar
        self.non = non
        self.spoof = spoof
        self.model_cond = model_cond
        self.seg_cond = seg_cond
        self.trial_cond = trial_cond
        self.model_cond_name = model_cond_name
        self.seg_cond_name = seg_cond_name
        self.trial_cond_name = trial_cond_name
        if (model_set is not None) and (seg_set is not None):
            self.validate()

    @property
    def num_models(self) -> int:
        return len(self.model_set)

    @property
    def num_tests(self) -> int:
        return len(self.seg_set)

    def copy(self) -> "TrialKey":
        """Makes a copy of the object"""
        return copy.deepcopy(self)

    def sort(self) -> None:
        """Sorts the object by model and test segment names."""
        self.model_set, m_idx = sort(self.model_set, return_index=True)
        self.seg_set, s_idx = sort(self.seg_set, return_index=True)
        ix = np.ix_(m_idx, s_idx)
        self.tar = self.tar[ix]
        self.non = self.non[ix]
        if self.spoof is not None:
            self.spoof = self.spoof[ix]
        if self.model_cond is not None:
            self.model_cond = self.model_cond[:, m_idx]
        if self.seg_cond is not None:
            self.seg_cond = self.seg_cond[:, s_idx]
        if self.trial_cond is not None:
            self.trial_cond = self.trial_cond[:, m_idx][:, :, s_idx]

    def save(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """Saves object to txt/h5 file.

        Args:
          file_path: File to write the list.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext in (".h5", ".hdf5"):
            self.save_h5(file_path)
        elif ext in ("", ".txt"):
            self.save_txt(file_path)
        else:
            self.save_table(file_path, sep)

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
            trial_mask = self.tar.astype("int8") - self.non.astype("int8")
            if self.spoof is not None:
                trial_mask -= 2 * self.spoof.astype("int8")
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

    def save_txt(self, file_path: PathLike) -> None:
        """Saves object to txt file.

        Args:
          file_path: File to write the list.
        """
        with open(file_path, "w") as f:
            idx = (self.tar.T).nonzero()
            for item in zip(idx[0], idx[1]):
                f.write(
                    "%s %s target\n" % (self.model_set[item[1]], self.seg_set[item[0]])
                )
            idx = (self.non.T).nonzero()
            for item in zip(idx[0], idx[1]):
                f.write(
                    "%s %s nontarget\n"
                    % (self.model_set[item[1]], self.seg_set[item[0]])
                )
            if self.spoof is not None:
                idx = (self.spoof.T).nonzero()
                for item in zip(idx[0], idx[1]):
                    f.write(
                        "%s %s spoof\n"
                        % (self.model_set[item[1]], self.seg_set[item[0]])
                    )

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
            # added in ASVSpoof 2024 to account for spoofing trials
            if self.spoof is not None:
                non = np.logical_or(self.non, self.spoof)
            else:
                non = self.non

            I, J = np.logical_or(self.tar, non).nonzero()
            for i, j in zip(I, J):
                target_type = (
                    "target"
                    if self.tar[i, j]
                    else ("nontarget" if self.non[i, j] else "spoof")
                )
                f.write(
                    f"{self.model_set[i]}{sep}{self.seg_set[j]}{sep}{target_type}\n"
                )

    @classmethod
    def load(cls, file_path: PathLike, sep: Optional[str] = None) -> "TrialKey":
        """Loads object from txt/h5 file

        Args:
          file_path: File to read the list.

        Returns:
          TrialKey object.
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
    def load_h5(cls, file_path: PathLike) -> "TrialKey":
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
            # added to account for spoofing trials in ASVSpoof 2024
            spoof = (trial_mask < -1).astype("bool")
            if np.any(spoof):
                trial_mask[spoof] = 0
            else:
                spoof = None
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
            spoof,
            model_cond,
            seg_cond,
            trial_cond,
            model_cond_name,
            seg_cond_name,
            trial_cond_name,
        )

    @classmethod
    def load_txt(cls, file_path: PathLike) -> "TrialKey":
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
        tar = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        non = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        if np.any(is_spoof):
            spoof = np.zeros((len(model_set), len(seg_set)), dtype="bool")
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
        return cls(model_set, seg_set, tar, non, spoof)

    @classmethod
    def load_table(cls, file_path: PathLike, sep: Optional[str] = None) -> "TrialKey":
        """Loads object from pandas table file

        Args:
          file_path: File to read the list.

        Returns:
          TrialKey object.
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
        tar = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        non = np.zeros((len(model_set), len(seg_set)), dtype="bool")
        if np.any(is_spoof):
            spoof = np.zeros((len(model_set), len(seg_set)), dtype="bool")
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
        return cls(model_set, seg_set, tar, non, spoof)

    @classmethod
    def merge(cls, key_list: List["TrialKey"]) -> "TrialKey":
        """Merges several key objects.

        Args:
          key_list: List of TrialKey objects.

        Returns:
          Merged TrialKey object.
        """
        if len(key_list) == 0:
            raise ValueError("key_list must contain at least one TrialKey")
        if len(key_list) == 1:
            return key_list[0].copy()

        def _check_consistent_presence(name: str, present: List[bool]) -> None:
            if any(present) and not all(present):
                raise ValueError(f"Cannot merge TrialKey with mixed {name} presence")

        def _check_equal_names(name: str, names: List[Optional[np.ndarray]]) -> None:
            present = [n is not None for n in names]
            if any(present) and not all(present):
                raise ValueError(f"Cannot merge TrialKey with mixed {name} presence")
            if all(present):
                ref = names[0]
                for n in names[1:]:
                    if not np.array_equal(ref, n):
                        raise ValueError(
                            f"All TrialKey objects must have identical {name}"
                        )

        has_spoof = [k.spoof is not None for k in key_list]
        has_model_cond = [k.model_cond is not None for k in key_list]
        has_seg_cond = [k.seg_cond is not None for k in key_list]
        has_trial_cond = [k.trial_cond is not None for k in key_list]

        _check_consistent_presence("spoof", has_spoof)
        _check_consistent_presence("model_cond", has_model_cond)
        _check_consistent_presence("seg_cond", has_seg_cond)
        _check_consistent_presence("trial_cond", has_trial_cond)

        use_spoof = all(has_spoof)
        use_model_cond = all(has_model_cond)
        use_seg_cond = all(has_seg_cond)
        use_trial_cond = all(has_trial_cond)

        if use_model_cond:
            num_model_cond = key_list[0].model_cond.shape[0]
            for k in key_list[1:]:
                if k.model_cond.shape[0] != num_model_cond:
                    raise ValueError(
                        "All TrialKey objects must have the same number of model conditions"
                    )
            _check_equal_names(
                "model_cond_name", [k.model_cond_name for k in key_list]
            )
            model_cond_name = key_list[0].model_cond_name
        else:
            model_cond_name = None

        if use_seg_cond:
            num_seg_cond = key_list[0].seg_cond.shape[0]
            for k in key_list[1:]:
                if k.seg_cond.shape[0] != num_seg_cond:
                    raise ValueError(
                        "All TrialKey objects must have the same number of segment conditions"
                    )
            _check_equal_names("seg_cond_name", [k.seg_cond_name for k in key_list])
            seg_cond_name = key_list[0].seg_cond_name
        else:
            seg_cond_name = None

        if use_trial_cond:
            num_trial_cond = key_list[0].trial_cond.shape[0]
            for k in key_list[1:]:
                if k.trial_cond.shape[0] != num_trial_cond:
                    raise ValueError(
                        "All TrialKey objects must have the same number of trial conditions"
                    )
            _check_equal_names(
                "trial_cond_name", [k.trial_cond_name for k in key_list]
            )
            trial_cond_name = key_list[0].trial_cond_name
        else:
            trial_cond_name = None

        num_key = len(key_list)
        model_set = key_list[0].model_set
        seg_set = key_list[0].seg_set
        tar = key_list[0].tar.copy()
        non = key_list[0].non.copy()
        spoof = key_list[0].spoof.copy() if use_spoof else None
        model_cond = key_list[0].model_cond.copy() if use_model_cond else None
        seg_cond = key_list[0].seg_cond.copy() if use_seg_cond else None
        trial_cond = key_list[0].trial_cond.copy() if use_trial_cond else None

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
            if use_spoof:
                spoof_1 = np.zeros(shape, dtype="bool")
                spoof_1[ix_a] = spoof[ix_b]
            if use_model_cond:
                model_cond_1 = np.zeros((num_model_cond, shape[0]), dtype="bool")
                model_cond_1[:, mi_a] = model_cond[:, mi_b]
            if use_seg_cond:
                seg_cond_1 = np.zeros((num_seg_cond, shape[1]), dtype="bool")
                seg_cond_1[:, si_a] = seg_cond[:, si_b]
            if use_trial_cond:
                trial_cond_1 = np.zeros((num_trial_cond, *shape), dtype="bool")
                cond_idx = np.arange(num_trial_cond)
                trial_cond_1[np.ix_(cond_idx, mi_a, si_a)] = trial_cond[
                    np.ix_(cond_idx, mi_b, si_b)
                ]

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
            if use_spoof:
                spoof_2 = np.zeros(shape, dtype="bool")
                spoof_2[ix_a] = key_i.spoof[ix_b]
            if use_model_cond:
                model_cond_2 = np.zeros((num_model_cond, shape[0]), dtype="bool")
                model_cond_2[:, mi_a] = key_i.model_cond[:, mi_b]
            if use_seg_cond:
                seg_cond_2 = np.zeros((num_seg_cond, shape[1]), dtype="bool")
                seg_cond_2[:, si_a] = key_i.seg_cond[:, si_b]
            if use_trial_cond:
                trial_cond_2 = np.zeros((num_trial_cond, *shape), dtype="bool")
                cond_idx = np.arange(num_trial_cond)
                trial_cond_2[np.ix_(cond_idx, mi_a, si_a)] = key_i.trial_cond[
                    np.ix_(cond_idx, mi_b, si_b)
                ]

            model_set = new_model_set
            seg_set = new_seg_set
            tar = np.logical_or(tar_1, tar_2)
            non = np.logical_or(non_1, non_2)
            if use_spoof:
                spoof = np.logical_or(spoof_1, spoof_2)
            if use_model_cond:
                model_cond = np.logical_or(model_cond_1, model_cond_2)
            if use_seg_cond:
                seg_cond = np.logical_or(seg_cond_1, seg_cond_2)
            if use_trial_cond:
                trial_cond = np.logical_or(trial_cond_1, trial_cond_2)

        return cls(
            model_set,
            seg_set,
            tar,
            non,
            spoof,
            model_cond,
            seg_cond,
            trial_cond,
            model_cond_name,
            seg_cond_name,
            trial_cond_name,
        )

    def filter(
        self,
        model_set: StrArrayLike,
        seg_set: StrArrayLike,
        keep: bool = True,
        raise_missing: bool = True,
    ) -> "TrialKey":
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

        return TrialKey(
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
    ) -> "TrialKey":
        """Removes elements from TrialKey object.

        Args:
          model_set: List of models to keep or remove.
          keep: If True, we keep the elements in model_set/seg_set,
                if False, we remove the elements in model_set/seg_set.

        Returns:
          Filtered TrialKey object.
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

        return TrialKey(
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
        self,
        model_idx: int,
        num_model_parts: int,
        seg_idx: int,
        num_seg_parts: int,
    ) -> "TrialKey":
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

        return TrialKey(
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
        mask = np.logical_or(self.tar, self.non)
        if self.spoof is not None:
            mask = np.logical_or(mask, self.spoof)
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
            self.tar = np.zeros(shape, dtype="bool")
            self.non = np.zeros(shape, dtype="bool")
        else:
            if self.tar.shape != shape:
                raise ValueError(f"tar shape {self.tar.shape} does not match {shape}")
            if self.non.shape != shape:
                raise ValueError(f"non shape {self.non.shape} does not match {shape}")
        if self.spoof is not None:
            if self.spoof.shape != shape:
                raise ValueError(f"spoof shape {self.spoof.shape} does not match {shape}")
        if np.any(np.logical_and(self.tar, self.non)):
            raise ValueError("tar and non overlap")
        if self.spoof is not None:
            if np.any(np.logical_and(self.tar, self.spoof)):
                raise ValueError("tar and spoof overlap")
            if np.any(np.logical_and(self.non, self.spoof)):
                raise ValueError("non and spoof overlap")

        if self.model_cond is not None:
            if self.model_cond.shape[1] != shape[0]:
                raise ValueError(
                    "model_cond second dimension must match number of models"
                )
        if self.seg_cond is not None:
            if self.seg_cond.shape[1] != shape[1]:
                raise ValueError(
                    "seg_cond second dimension must match number of segments"
                )
        if self.trial_cond is not None:
            if self.trial_cond.shape[1:] != shape:
                raise ValueError(
                    "trial_cond trailing dimensions must match (num_models, num_segments)"
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

    def __eq__(self, other: object) -> bool:
        """Equal operator"""
        if not isinstance(other, TrialKey):
            return False

        eq = self.model_set.shape == other.model_set.shape
        eq = eq and np.all(self.model_set == other.model_set)
        eq = eq and (self.seg_set.shape == other.seg_set.shape)
        eq = eq and np.all(self.seg_set == other.seg_set)
        eq = eq and np.all(self.tar == other.tar)
        eq = eq and np.all(self.non == other.non)
        eq = eq and (
            self.spoof is None
            and other.spoof is None
            or np.all(self.spoof == other.spoof)
        )

        eq = eq and ((self.model_cond is None) == (other.model_cond is None))
        eq = eq and ((self.seg_cond is None) == (other.seg_cond is None))
        eq = eq and ((self.trial_cond is None) == (other.trial_cond is None))
        if self.model_cond is not None:
            eq = eq and np.all(self.model_cond == other.model_cond)
        if self.seg_cond is not None:
            eq = eq and np.all(self.seg_cond == other.seg_cond)
        if self.trial_cond is not None:
            eq = eq and np.all(self.trial_cond == other.trial_cond)

        eq = eq and ((self.model_cond_name is None) == (other.model_cond_name is None))
        eq = eq and ((self.seg_cond_name is None) == (other.seg_cond_name is None))
        eq = eq and ((self.trial_cond_name is None) == (other.trial_cond_name is None))

        if self.model_cond_name is not None:
            eq = eq and np.all(self.model_cond_name == other.model_cond_name)
        if self.seg_cond_name is not None:
            eq = eq and np.all(self.seg_cond_name == other.seg_cond_name)
        if self.trial_cond_name is not None:
            eq = eq and np.all(self.trial_cond_name == other.trial_cond_name)
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
        assert key1 == key3

        file_txt = "test.txt"
        key3.tar[0, :] = True
        key3.non[:, 0] = True
        key3.save(file_txt)
        key2 = TrialKey.load(file_txt)
        assert key3 == key2
