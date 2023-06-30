"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import yaml

from .class_info import ClassInfo
from .feature_set import FeatureSet
from .misc import PathLike
from .recording_set import RecordingSet
from .segment_set import SegmentSet
from .enrollment_map import EnrollmentMap
from .trial_key import TrialKey
from .trial_ndx import TrialNdx
from .sparse_trial_key import SparseTrialKey


class Dataset:
    """Class that contains all objects
    (segments, recordings, features, class_infos) that
    conform a dataset

    Attributes:
      segments:     SegmentSet object or path to it.
      classes:      Dictionary of ClassInfo objects or paths to then
      recordings:   Dictionary of RecordingSet objects or paths to then
      features:     Dictionary of FeatureSet objects or paths to then
      enrollments:  Dictionary of EnrollmentMap objects or paths to then
      trials:       Dictionary of TrialKey/TrialNdx/SparseTrialKey objects
        or paths to then
      sparse_trials: load trial keys using the SparseTrialKey class instead
          of TrialKey class.
      table_sep:    Column separator when reading/writting tables

    """

    def __init__(
        self,
        segments: Union[SegmentSet, PathLike],
        classes: Optional[Dict[str, Union[ClassInfo, PathLike]]] = None,
        recordings: Optional[Dict[str, Union[RecordingSet, PathLike]]] = None,
        features: Optional[Dict[str, Union[FeatureSet, PathLike]]] = None,
        enrollments: Optional[Dict[str, Union[EnrollmentMap, PathLike]]] = None,
        trials: Optional[
            Dict[str, Union[TrialKey, TrialNdx, SparseTrialKey, PathLike]]
        ] = None,
        sparse_trials: bool = False,
        table_sep: Optional[str] = None,
    ):

        if isinstance(segments, SegmentSet):
            self._segments = segments
            self._segments_path = None
        else:
            assert isinstance(segments, (str, Path))
            self._segments = None
            self._segments_path = Path(segments)

        self._classes, self._classes_paths = self._parse_dict_args(classes, ClassInfo)

        self._recordings, self._recordings_paths = self._parse_dict_args(
            recordings, RecordingSet
        )

        self._features, self._features_paths = self._parse_dict_args(
            features, FeatureSet
        )
        self._enrollments, self._enrollments_paths = self._parse_dict_args(
            enrollments,
            EnrollmentMap,
        )
        self._trials, self._trials_paths = self._parse_dict_args(
            trials,
            (TrialKey, TrialNdx, SparseTrialKey),
        )

        self.sparse_trials = sparse_trials
        self.table_sep = table_sep

    def _parse_dict_args(self, data, types):
        if data is None:
            return None, None

        assert isinstance(data, dict)
        objects = {k: (v if isinstance(v, types) else None) for k, v in data.items()}
        paths = {
            k: (v if isinstance(v, (str, Path)) else None) for k, v in data.items()
        }

        return objects, paths

    def clone(self):
        return deepcopy(self)

    def segments(self, keep_loaded: bool = True):
        if self._segments is None:
            assert self._segments_path is not None
            segments = SegmentSet.load(self._segments_path, sep=self.table_sep)
            if keep_loaded:
                self._segments = segments
            return segments

        return self._segments

    def recordings_value(self, key: str, keep_loaded: bool = True):
        if self._recordings[key] is None:
            assert self._recordings_paths[key] is not None
            recordings = RecordingSet.load(
                self._recordings_paths[key], sep=self.table_sep
            )
            if keep_loaded:
                self._recordings[key] = recordings
            return recordings

        return self._recordings[key]

    def features_value(self, key: str, keep_loaded: bool = True):
        if self._features[key] is None:
            assert self._features_paths[key] is not None
            features = FeatureSet.load(self._features_paths[key], sep=self.table_sep)
            if keep_loaded:
                self._features[key] = features
            return features

        return self._features[key]

    def classes_value(self, key: str, keep_loaded: bool = True):
        if self._classes[key] is None:
            assert self._classes_paths[key] is not None
            classes = ClassInfo.load(self._classes_paths[key], self.table_sep)
            if keep_loaded:
                self._classes[key] = classes
            return classes

        return self._classes[key]

    def enrollments_value(self, key: str, keep_loaded: bool = True):
        if self._enrollments[key] is None:
            assert self._enrollments_paths[key] is not None
            enrollments = EnrollmentMap.load(
                self._enrollments_paths[key], sep=self.table_sep
            )
            if keep_loaded:
                self._enrollments[key] = enrollments
            return enrollments

        return self._enrollments[key]

    def trials_value(self, key: str, keep_loaded: bool = True):
        if self._trials[key] is None:
            assert self._trials_paths[key] is not None
            try:
                if self.sparse_trials:
                    trials = SparseTrialKey.load(self._trials_paths[key])
                else:
                    trials = TrialKey.load(self._trials_paths[key])
            except:
                trials = TrialNdx.load(self._trials_paths[key])

            if keep_loaded:
                self._trials[key] = trials
            return trials

        return self._trials[key]

    def recordings(self, keep_loaded: bool = True):
        if self._recordings is None:
            yield from ()
        else:
            for key in self._recordings.keys():
                yield key, self.recordings_value(key, keep_loaded)

    def features(self, keep_loaded: bool = True):
        if self._features is None:
            yield from ()
        else:
            for key in self._features.keys():
                yield key, self.features_value(key, keep_loaded)

    def classes(self, keep_loaded: bool = True):
        if self._classes is None:
            yield from ()
        else:
            for key in self._classes.keys():
                yield key, self.classes_value(key, keep_loaded)

    def enrollments(self, keep_loaded: bool = True):
        if self._enrollments is None:
            yield from ()
        else:
            for key in self._enrollments.keys():
                yield key, self.enrollments_value(key, keep_loaded)

    def trials(self, keep_loaded: bool = True):
        if self._trials is None:
            yield from ()
        else:
            for key in self._trials.keys():
                yield key, self.trials_value(key, keep_loaded)

    # def add_recordings(self, recordings: Dict[str, Union[RecordingSet, PathLike]]):
    #     recordings, recordings_paths = self._parse_dict_args(recordings, RecordingSet)
    #     if self._recordings is None:
    #         self._recordings = self._recordings_paths = {}
    #     self._recordings.update(recordings)
    #     self._recordings_paths.update(recordings_paths)

    # def add_features(self, features: Dict[str, Union[FeatureSet, PathLike]]):
    #     features, features_paths = self._parse_dict_args(features, FeatureSet)
    #     if self._features is None:
    #         self._features = self._features_paths = {}
    #     self._features.update(features)
    #     self._features_paths.update(features_paths)

    # def add_classes(self, classes: Dict[str, Union[ClassInfo, PathLike]]):
    #     classes, classes_paths = self._parse_dict_args(classes, ClassInfo)
    #     if self._classes is None:
    #         self._classes = self._classes_paths = {}
    #     self._classes.update(classes)
    #     self._classes_paths.update(classes_paths)

    # def add_enrollments(self, enrollments: Dict[str, Union[EnrollmentMap, PathLike]]):
    #     enrollments, enrollments_paths = self._parse_dict_args(
    #         enrollments,
    #         EnrollmentMap,
    #     )
    #     if self._enrollments is None:
    #         self._enrollments = self._enrollments_paths = {}
    #     self._enrollments.update(enrollments)
    #     self._enrollments_paths.update(enrollments_paths)

    # def add_trials(
    #     self, trials: Dict[str, Union[TrialKey, TrialNdx, SparseTrialKey, PathLike]]
    # ):
    #     trials, trials_paths = self._parse_dict_args(
    #         trials,
    #         (TrialKey, TrialNdx, SparseTrialKey),
    #     )
    #     if self._trials is None:
    #         self._trials = self._trials_paths = {}
    #     self._trials.update(trials)
    #     self._trials_paths.update(trials_paths)

    @staticmethod
    def resolve_dataset_path(dataset_path):
        dataset_path = Path(dataset_path)
        ext = dataset_path.suffix
        if ext in [".yaml", "yml"]:
            dataset_file = dataset_path
            dataset_dir = dataset_path.parent
        else:
            dataset_file = dataset_path / "dataset.yaml"
            dataset_dir = dataset_path

        return dataset_dir, dataset_file

    @staticmethod
    def resolve_file_path(dataset_dir, file_path):
        dataset_dir = Path(dataset_dir)
        file_path = Path(file_path)
        if file_path.is_file():
            return file_path

        return dataset_dir / file_path

    def save(
        self,
        dataset_path: PathLike,
        update_paths: bool = True,
        table_sep: Optional[str] = None,
        force_save_all: bool = False,
    ):
        """Saves the dataset to disk.

        Args:
          dataset_path: str/Path indicating directory
            to save the dataset or .yaml file to save
            the dataset info.
          update_paths: whether to update the file_paths in the
            data structures in the DataSet object
          force_save_all: forces saving all tables even if they haven't changed,
                          otherwise, it only saves tables loaded in memory
                          and those that are not in the datadirectory
        """
        if force_save_all:
            self.save_all(dataset_path, update_paths, table_sep)
        else:
            self.save_changed(dataset_path, update_paths, table_sep)

    def save_changed(
        self,
        dataset_path: PathLike,
        update_paths: bool = True,
        table_sep: Optional[str] = None,
        force_save_all: bool = False,
    ):
        """Saves the tables that change in disk or tables
           that are not in the ouput directory.

        Args:
          dataset_path: str/Path indicating directory
            to save the dataset or .yaml file to save
            the dataset info.
          update_paths: whether to update the file_paths in the
            data structures in the DataSet object
        """
        table_sep = self.table_sep if table_sep is None else table_sep
        if update_paths:
            self.table_sep = table_sep

        table_ext = ".tsv" if table_sep == "\t" else ".csv"
        dataset_dir, dataset_file = Dataset.resolve_dataset_path(dataset_path)
        dataset = {}
        file_name = f"segments{table_ext}"
        dataset["segments"] = file_name
        file_path = dataset_dir / file_name
        if (
            self._segments is not None
            or file_path != self._segments_path
            or not file_path.exists()
        ):
            self.segments(keep_loaded=False).save(file_path, sep=table_sep)
            if update_paths:
                self._segments_path = file_path

        if self._recordings is not None:
            file_names = {}
            for k in self._recordings.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._recordings[k] is not None
                    or file_path != self._recordings_paths[k]
                    or not file_path.exists()
                ):
                    v = self.recordings_value(k, keep_loaded=False)
                    v.save(file_path, sep=table_sep)
                    if update_paths:
                        self._recordings_paths[k] = file_path

            if file_names:
                dataset["recordings"] = file_names

        if self._features is not None:
            file_names = {}
            for k in self._features.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._features[k] is not None
                    or file_path != self._features_paths[k]
                    or not file_path.exists()
                ):
                    v = self.features_value(k, keep_loaded=False)
                    v.save(file_path, sep=table_sep)
                    if update_paths:
                        self._features_paths[k] = file_path

            if file_names:
                dataset["features"] = file_names

        if self._classes is not None:
            file_names = {}
            for k in self._classes.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._classes[k] is not None
                    or file_path != self._classes_paths[k]
                    or not file_path.exists()
                ):
                    v = self.classes_value(k, keep_loaded=False)
                    v.save(file_path, sep=table_sep)
                    if update_paths:
                        self._classes_paths[k] = file_path

            if file_names:
                dataset["classes"] = file_names

        if self._enrollments is not None:
            file_names = {}
            for k in self._enrollments.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._enrollments[k] is not None
                    or file_path != self._enrollments_paths[k]
                    or not file_path.exists()
                ):
                    v = self.enrollments_value(k, keep_loaded=False)
                    v.save(file_path, sep=table_sep)
                    if update_paths:
                        self._enrollments_paths[k] = file_path

            if file_names:
                dataset["enrollments"] = file_names

        if self._trials is not None:
            file_names = {}
            for k in self._trials.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._trials[k] is not None
                    or file_path != self._trials_paths[k]
                    or not file_path.exists()
                ):
                    v = self.trials_value(k, keep_loaded=False)
                    v.save(file_path)
                    if update_paths:
                        self._trials_paths[k] = file_path

            if file_names:
                dataset["trials"] = file_names

        with open(dataset_file, "w") as f:
            yaml.dump(dataset, f)

    def save_all(
        self,
        dataset_path: PathLike,
        update_paths: bool = True,
        table_sep: Optional[str] = None,
    ):
        """Saves all the dataset objects.

        Args:
          dataset_path: str/Path indicating directory
            to save the dataset or .yaml file to save
            the dataset info.
          update_paths: whether to update the file_paths in the
            data structures in the DataSet object
        """
        table_sep = self.table_sep if table_sep is None else table_sep
        if update_paths:
            self.table_sep = table_sep

        table_ext = ".tsv" if table_sep == "\t" else ".csv"
        dataset_dir, dataset_file = Dataset.resolve_dataset_path(dataset_path)
        dataset = {}
        file_name = f"segments{table_ext}"
        dataset["segments"] = file_name
        file_path = dataset_dir / file_name
        self.segments(keep_loaded=False).save(file_path, sep=table_sep)
        if update_paths:
            self._segments_path = file_path

        file_names = {}
        for k, v in self.recordings(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path, sep=table_sep)
            if update_paths:
                self._recordings_paths[k] = file_path

        if file_names:
            dataset["recordings"] = file_names

        file_names = {}
        for k, v in self.features(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path, sep=table_sep)
            if update_paths:
                self._features_paths[k] = file_path

        if file_names:
            dataset["features"] = file_names

        file_names = {}
        for k, v in self.classes(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path, sep=table_sep)
            if update_paths:
                self._classes_paths[k] = file_path

        if file_names:
            dataset["classes"] = file_names

        file_names = {}
        for k, v in self.enrollments(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path, sep=table_sep)
            if update_paths:
                self._enrollments_paths[k] = file_path

        if file_names:
            dataset["enrollments"] = file_names

        file_names = {}
        for k, v in self.trials(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path)
            if update_paths:
                self._trials_paths[k] = file_path

        if file_names:
            dataset["trials"] = file_names

        with open(dataset_file, "w") as f:
            yaml.dump(dataset, f)

    def update_from_disk(self):
        self.segments()
        for k, v in self.recordings():
            pass

        for k, v in self.features():
            pass

        for k, v in self.classes():
            pass

        for k, v in self.enrollments():
            pass

        for k, v in self.trials():
            pass

    @classmethod
    def load(
        cls, dataset_path: PathLike, lazy: bool = True, sparse_trials: bool = False
    ):
        """Loads all the dataset objects.

        Args:
         dataset_path: str/Path indicating directory
          to save the dataset or .yaml file to save
          the dataset info.
         lazy: load data structures lazily when they are needed.
         sparse_trials: load trial keys using the SparseTrialKey class instead of TrialKey class

        """
        dataset_dir, dataset_file = Dataset.resolve_dataset_path(dataset_path)
        with open(dataset_file, "r") as f:
            dataset = yaml.safe_load(f)

        assert "segments" in dataset
        segments = Dataset.resolve_file_path(dataset_dir, dataset["segments"])
        classes = None
        recordings = None
        features = None
        enrollments = None
        trials = None
        if "classes" in dataset:
            classes = {}
            for k, v in dataset["classes"].items():
                classes[k] = Dataset.resolve_file_path(dataset_dir, v)

        if "recordings" in dataset:
            recordings = {}
            for k, v in dataset["recordings"].items():
                recordings[k] = Dataset.resolve_file_path(dataset_dir, v)

        if "features" in dataset:
            features = {}
            for k, v in dataset["features"].items():
                features[k] = Dataset.resolve_file_path(dataset_dir, v)

        if "enrollments" in dataset:
            enrollments = {}
            for k, v in dataset["enrollments"].items():
                enrollments[k] = Dataset.resolve_file_path(dataset_dir, v)

        if "trials" in dataset:
            trials = {}
            for k, v in dataset["trials"].items():
                trials[k] = Dataset.resolve_file_path(dataset_dir, v)

        dataset = cls(
            segments,
            classes,
            recordings,
            features,
            enrollments,
            trials,
            sparse_trials=sparse_trials,
        )
        if not lazy:
            dataset.update_from_disk()

        return dataset

    def add_features(self, features_name: str, features: Union[PathLike, FeatureSet]):
        if self._features is None:
            self._features = {}
            self._features_paths = {}

        if isinstance(features, (str, Path)):
            self._features[features_name] = None
            self._features_paths[features_name] = features
        elif isinstance(features, FeatureSet):
            self._features[features_name] = features
            self._features_paths[features_name] = None
        else:
            raise ValueError()

    def add_recordings(
        self,
        recordings_name: str,
        recordings: Union[PathLike, RecordingSet],
    ):
        if self._recordings is None:
            self._recordings = {}
            self._recordings_paths = {}

        if isinstance(features, (str, Path)):
            self._recordings[features_name] = None
            self._recordings_paths[recordings_name] = recordings
        elif isinstance(recordings, RecordingSet):
            self._recordings[recordings_name] = recordings
            self._recordings_paths[recordings_name] = None
        else:
            raise ValueError()

    def add_classes(self, classes_name: str, classes: Union[PathLike, ClassInfo]):
        if self._classes is None:
            self._classes = {}
            self._classes_paths = {}

        if isinstance(classes, (str, Path)):
            self._classes[features_name] = None
            self._classes_paths[classes_name] = classes
        elif isinstance(classes, ClassInfo):
            self._classes[classes_name] = classes
            self._classes_paths[classes_name] = None
        else:
            raise ValueError()

    def add_enrollments(
        self,
        enrollments_name: str,
        enrollments: Union[PathLike, EnrollmentMap],
    ):
        if self._enrollments is None:
            self._enrollments = {}
            self._enrollments_paths = {}

        if isinstance(enrollments, (str, Path)):
            self._enrollments[enrollments_name] = None
            self._enrollments_paths[enrollments_name] = enrollments
        elif isinstance(enrollments, EnrollmentMap):
            self._enrollments[enrollments_name] = enrollments
            self._enrollments_paths[enrollments_name] = None
        else:
            raise ValueError()

    def add_trials(
        self,
        trials_name: str,
        trials: Union[PathLike, TrialKey, TrialNdx, SparseTrialKey],
    ):
        if self._trials is None:
            self._trials = {}
            self._trials_paths = {}

        if isinstance(trials, (str, Path)):
            self._trials[features_name] = None
            self._trials_paths[trials_name] = trials
        elif isinstance(trials, (TrialKey, TrialNdx, SparseTrialKey)):
            self._trials[trials_name] = trials
            self._trials_paths[trials_name] = None
        else:
            raise ValueError()

    def remove_features(self, features_name: str):
        if self._features_paths[features_name] is not None:
            file_path = Path(self._features_paths[features_name])
            if file_path.is_file():
                file_path.unlink()

        del self._features[features_name]
        del self._features_paths[features_name]

    def remove_recordings(
        self,
        recordings_name: str,
    ):
        if self._recordingsr_paths[recordings_name] is not None:
            file_path = Path(self._recordings_paths[recordings_name])
            if file_path.is_file():
                file_path.unlink()

        del self._recordings[recordings_name]
        del self._recordings_paths[recordings_name]

    def remove_classes(self, classes_name: str):
        if self._classes_paths[classes_name] is not None:
            file_path = Path(self._classes_paths[classes_name])
            if file_path.is_file():
                file_path.unlink()

        del self._classes[classes_name]
        del self._classes_paths[classes_name]

    def remove_enrollments(
        self,
        enrollments_name: str,
    ):
        if self._enrollments_paths[enrollments_name] is not None:
            file_path = Path(self._enrollments_paths[enrollments_name])
            if file_path.is_file():
                file_path.unlink()

        del self._enrollments[enrollments_name]
        del self._enrollments_paths[enrollments_name]

    def remove_trials(
        self,
        trials_name: str,
    ):
        if self._trials_paths[trials_name] is not None:
            file_path = Path(self._trials_paths[trials_name])
            if file_path.is_file():
                file_path.unlink()

        del self._trials[trials_name]
        del self._trials_paths[trials_name]

    def set_segments(self, segments: Union[PathLike, SegmentSet]):
        if isinstance(segments, SegmentSet):
            self._segments = segments
        else:
            self._segments_path = segments

    def clean(self):
        rec_ids = self.segments().recording_ids()
        for k, table in self.recordings():
            table = table.loc[table["id"].isin(rec_ids)].copy()
            self._recordings[k] = RecordingSet(table)

        ids = self.segments()["id"].values
        for k, table in self.features():
            table = table.loc[table["id"].isin(ids)].copy()
            self._features[k] = FeatureSet(table)

        for k, table in self.classes():
            class_ids = self.segments()[k].unique()
            table = table[table["id"].isin(class_ids)].copy()
            self._classes[k] = ClassInfo(table)

        remove_keys = []
        for k, table in self.enrollments():
            table = table.loc[table["segmentid"].isin(ids)].copy()
            if len(table) > 0:
                self._enrollments[k] = EnrollmentMap(table)
            else:
                remove_keys.append(k)

        for k in remove_keys:
            self.remove_enrollments(k)

        remove_keys = []
        for k, key in self.trials():
            keep_ids = [cur_id for cur_id in key.seg_set if cur_id in ids]
            if keep_ids:
                key = key.filter(key.model_set, keep_ids, keep=True)
                self._trials[k] = key
            else:
                remove_keys.append(k)

        for k in remove_keys:
            self.remove_trials(k)

    def _split_into_trials_and_cohort(
        self,
        segments: SegmentSet,
        num_tar_trials: int,
        num_trial_speakers: int,
        seed: int,
    ):
        # select test speakers
        rng = np.random.RandomState(seed=seed)

        spks = segments["speaker"].unique()
        trial_spks = rng.choice(spks, size=(num_trial_speakers,), replace=False)
        snorm_segments = SegmentSet(segments[~segments["speaker"].isin(trial_spks)])

        trial_segments = segments[segments["speaker"].isin(trial_spks)]
        # solution of 2nd degree eq.
        # num_spks * n (n-1) /2 = num_trials
        num_segs_per_spk = int(
            math.ceil((1 + math.sqrt(1 + 8 * num_tar_trials // num_trial_speakers)) / 2)
        )

        n = num_trial_speakers * num_segs_per_spk
        seg_ids = rng.choice(trial_segments["id"], size=(n,), replace=False)
        trial_segments = SegmentSet(segments[segments["id"].isin(seg_ids)])
        seg_ids = trial_segments["id"].values
        class_ids = trial_segments["speaker"].values
        tar = np.zeros((n - 1, n), dtype=bool)
        non = np.zeros((n - 1, n), dtype=bool)

        ntar = 0
        nnon = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if class_ids[i] == class_ids[j]:
                    tar[i, j] = True
                else:
                    non[i, j] = True

        logging.info("Got ntar=%d and nnon=%d", tar.sum(), non.sum())
        trials = TrialKey(seg_ids[:-1], seg_ids, tar, non)
        df_enr = pd.DataFrame({"id": seg_ids[:-1], "segmentid": seg_ids[:-1]})
        enrollments = EnrollmentMap(df_enr)
        return trials, enrollments, snorm_segments

    def split_into_trials_and_cohort(
        self,
        num_1k_tar_trials: int,
        num_trial_speakers: int,
        intra_gender: bool = True,
        trials_name="trials_qmf",
        seed=1123,
    ):
        """When training quality measure fusion in, e.g., VoxCeleb recipe.
        We split the data into 2 parts:
            1) used to calculate SV scores to train the fusion
            2) cohort used to calculate the S-Norm parameters used in the QMF.

        The trials_file will be stored in the current dataset
        A new dataset is created with only the cohort speakers

        Args:
          num_1k_tar_trials: num of 1000 target trials.
          num_trial_speakers: number of spks used to create trials.
          intra_gender: if True, no cross gender trials are done.

        Returns:
          Dataset used for trials with trial list.
          Dataset used for cohort.
        """
        num_tar_trials = num_1k_tar_trials * 1000
        if intra_gender:
            num_tar_trials = num_tar_trials // 2
            num_trial_speakers = num_trial_speakers // 2
            segments = self.segments()
            segments_male = SegmentSet(segments[segments["gender"] == "m"])
            segments_female = SegmentSet(segments[segments["gender"] == "f"])
            trials_male, enroll_male, cohort_male = self._split_into_trials_and_cohort(
                segments_male,
                num_tar_trials,
                num_trial_speakers,
                seed,
            )
            (
                trials_female,
                enroll_female,
                cohort_female,
            ) = self._split_into_trials_and_cohort(
                segments_female,
                num_tar_trials,
                num_trial_speakers,
                seed,
            )
            trials = TrialKey.merge([trials_male, trials_female])
            enroll = EnrollmentMap.cat([enroll_male, enroll_female])
            cohort = SegmentSet.cat([cohort_male, cohort_female])
        else:
            segments = self.segments()
            trials, enroll, cohort = self._split_into_trials_and_cohort(
                segments,
                num_tar_trials,
                num_trial_speakers,
                seed,
            )

        dataset_trials = self.clone()
        segments = self.segments()
        trials_segments = SegmentSet(segments.loc[segments["id"].isin(trials.seg_set)])
        dataset_trials.set_segments(trials_segments)
        dataset_trials.add_trials("trials", trials)
        dataset_trials.add_enrollments("enrollments", enroll)
        dataset_trials.clean()

        dataset_cohort = self.clone()
        dataset_cohort.set_segments(cohort)
        dataset_cohort.clean()

        return dataset_trials, dataset_cohort
