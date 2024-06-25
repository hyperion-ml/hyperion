"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import lhotse
import numpy as np
import pandas as pd
import yaml

from .class_info import ClassInfo
from .enrollment_map import EnrollmentMap
from .feature_set import FeatureSet
from .info_table import InfoTable
from .misc import PathLike
from .recording_set import RecordingSet
from .segment_set import SegmentSet
from .sparse_trial_key import SparseTrialKey
from .trial_key import TrialKey
from .trial_ndx import TrialNdx


class HypDataset:
    """Class that contains all objects
    (segments, recordings, features, class_infos) that
    conform a dataset

    Attributes:
      segments:     SegmentSet object or path to it.
      classes:      Dictionary of ClassInfo objects or paths to then
      recordings:   RecordingSet object or paths to then
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
        recordings: Optional[Union[RecordingSet, PathLike]] = None,
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
        if recordings is not None:
            if isinstance(recordings, RecordingSet):
                self._recordings = recordings
                self._recordings_path = None
            else:
                assert isinstance(recordings, (str, Path))
                self._recordings = None
                self._recordings_path = Path(recordings)

        # self._recordings, self._recordings_paths = self._parse_dict_args(
        #     recordings, RecordingSet
        # )

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
        self._files_to_delete = []
        self.fix_segments_dtypes()

    def fix_segments_dtypes(self):
        if self._segments is not None:
            self._fix_segments_dtypes(self._segments)

    def _fix_segments_dtypes(self, segments):
        # ids in class_infos should be strings in segment set columns
        for k in self.classes_keys():
            segments.convert_col_to_str(k)

    def get_dataset_files(self):
        file_paths = []
        for file_path in [self._segments_path, self._recordings_path]:
            if file_path is not None:
                file_paths.append(file_path)

        for path_dict in [
            self._features_paths,
            self._enrollments_paths,
            self._trials_paths,
        ]:
            if path_dict is None:
                continue
            for k, v in path_dict.items():
                file_paths.append(v)

        return file_paths

    def _delete_files(self, dataset_dir):
        if not self._files_to_delete:
            return

        dataset_files = self.get_dataset_files()
        for file_path in self._files_to_delete:
            file_path = Path(file_path)
            # if the file has been added again we don't delete
            if file_path in dataset_files:
                continue

            # if we are saving the dataset to another location
            # we don't delete the one in the original
            if file_path.parent == dataset_dir and file_path.is_file():
                file_path.unlink()

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
            self._fix_segments_dtypes(segments)
            if keep_loaded:
                self._segments = segments
            return segments

        return self._segments

    def __len__(self):
        return len(self.segments())

    def recordings(self, keep_loaded: bool = True):
        if self._recordings is None:
            assert self._recordings_path is not None
            recordings = RecordingSet.load(self._recordings_path, sep=self.table_sep)
            if keep_loaded:
                self._recordings = recordings
            return recordings

        return self._recordings

    # def recordings_value(self, key: str, keep_loaded: bool = True):
    #     if self._recordings[key] is None:
    #         assert self._recordings_paths[key] is not None
    #         recordings = RecordingSet.load(
    #             self._recordings_paths[key], sep=self.table_sep
    #         )
    #         if keep_loaded:
    #             self._recordings[key] = recordings
    #         return recordings

    #     return self._recordings[key]

    def features_keys(self):
        if self._features is not None:
            return self._features.keys()
        elif self._features_paths is not None:
            return self._features_paths.keys()
        else:
            return {}

    def features_value(self, key: str, keep_loaded: bool = True):
        if self._features[key] is None:
            assert self._features_paths[key] is not None
            features = FeatureSet.load(self._features_paths[key], sep=self.table_sep)
            if keep_loaded:
                self._features[key] = features
            return features

        return self._features[key]

    def classes_keys(self):
        if self._classes is not None:
            return self._classes.keys()
        elif self._classes_paths is not None:
            return self._classes_paths.keys()
        else:
            return {}

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

    # def recordings(self, keep_loaded: bool = True):
    #     if self._recordings is None:
    #         yield from ()
    #     else:
    #         for key in self._recordings.keys():
    #             yield key, self.recordings_value(key, keep_loaded)

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
        dataset_dir, dataset_file = HypDataset.resolve_dataset_path(dataset_path)
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

        file_name = f"recordings{table_ext}"
        dataset["recordings"] = file_name
        file_path = dataset_dir / file_name
        if (
            self._recordings is not None
            or file_path != self._recordings_path
            or not file_path.exists()
        ):
            self.recordings(keep_loaded=False).save(file_path, sep=table_sep)
            if update_paths:
                self._recordings_path = file_path

        # if self._recordings is not None:
        #     file_names = {}
        #     for k in self._recordings.keys():
        #         file_name = k + table_ext
        #         file_names[k] = file_name
        #         file_path = dataset_dir / file_name
        #         if (
        #             self._recordings[k] is not None
        #             or file_path != self._recordings_paths[k]
        #             or not file_path.exists()
        #         ):
        #             v = self.recordings_value(k, keep_loaded=False)
        #             v.save(file_path, sep=table_sep)
        #             if update_paths:
        #                 self._recordings_paths[k] = file_path

        #     if file_names:
        #         dataset["recordings"] = file_names

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

        self._delete_files(dataset_dir)

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
        dataset_dir, dataset_file = HypDataset.resolve_dataset_path(dataset_path)
        dataset = {}
        file_name = f"segments{table_ext}"
        dataset["segments"] = file_name
        file_path = dataset_dir / file_name
        self.segments(keep_loaded=False).save(file_path, sep=table_sep)
        if update_paths:
            self._segments_path = file_path

        file_name = f"recordings{table_ext}"
        dataset["recordings"] = file_name
        file_path = dataset_dir / file_name
        self.recordings(keep_loaded=False).save(file_path, sep=table_sep)
        if update_paths:
            self._recordings_path = file_path

        # file_names = {}
        # for k, v in self.recordings(keep_loaded=False):
        #     file_name = k + table_ext
        #     file_names[k] = file_name
        #     file_path = dataset_dir / file_name
        #     v.save(file_path, sep=table_sep)
        #     if update_paths:
        #         self._recordings_paths[k] = file_path

        # if file_names:
        #     dataset["recordings"] = file_names

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

        self._delete_files(dataset_dir)

    def update_from_disk(self):
        self.segments()
        self.recordings()

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
        dataset_dir, dataset_file = HypDataset.resolve_dataset_path(dataset_path)
        with open(dataset_file, "r") as f:
            dataset = yaml.safe_load(f)

        assert "segments" in dataset
        segments = HypDataset.resolve_file_path(dataset_dir, dataset["segments"])
        classes = None
        recordings = None
        features = None
        enrollments = None
        trials = None
        if "classes" in dataset:
            classes = {}
            for k, v in dataset["classes"].items():
                classes[k] = HypDataset.resolve_file_path(dataset_dir, v)

        if "recordings" in dataset:
            recordings = HypDataset.resolve_file_path(
                dataset_dir, dataset["recordings"]
            )
            # recordings = {}
            # for k, v in dataset["recordings"].items():
            #     recordings[k] = HypDataset.resolve_file_path(dataset_dir, v)

        if "features" in dataset:
            features = {}
            for k, v in dataset["features"].items():
                features[k] = HypDataset.resolve_file_path(dataset_dir, v)

        if "enrollments" in dataset:
            enrollments = {}
            for k, v in dataset["enrollments"].items():
                enrollments[k] = HypDataset.resolve_file_path(dataset_dir, v)

        if "trials" in dataset:
            trials = {}
            for k, v in dataset["trials"].items():
                trials[k] = HypDataset.resolve_file_path(dataset_dir, v)

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

    def set_segments(
        self,
        segments: Union[PathLike, SegmentSet],
    ):
        if isinstance(segments, (str, Path)):
            self._segments = None
            self._segments_path = segments
        elif isinstance(segments, SegmentSet):
            self._segments = segments
            self._segments_path = None
        else:
            raise ValueError()

    def set_recordings(
        self,
        recordings: Union[PathLike, RecordingSet],
        update_seg_durs: bool = False,
    ):
        if isinstance(recordings, (str, Path)):
            self._recordings = None
            self._recordings_path = Path(recordings)
        elif isinstance(recordings, RecordingSet):
            self._recordings = recordings
            self._recordings_path = None
        else:
            raise ValueError()

        if update_seg_durs:
            rec_ids = self.segments(keep_loaded=True).recordings()
            self.segments()["duration"] = self.recordings().loc[rec_ids, "duration"]

    def add_classes(self, classes_name: str, classes: Union[PathLike, ClassInfo]):
        if self._classes is None:
            self._classes = {}
            self._classes_paths = {}

        if isinstance(classes, (str, Path)):
            self._classes[classes_name] = None
            self._classes_paths[classes_name] = Path(classes)
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
            self._enrollments_paths[enrollments_name] = Path(enrollments)
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
            self._trials[trials_name] = None
            self._trials_paths[trials_name] = Path(trials)
        elif isinstance(trials, (TrialKey, TrialNdx, SparseTrialKey)):
            self._trials[trials_name] = trials
            self._trials_paths[trials_name] = None
        else:
            raise ValueError()

    def remove_features(self, features_name: str):
        if self._features_paths[features_name] is not None:
            self._files_to_delete.append(self._features_paths[features_name])

        del self._features[features_name]
        del self._features_paths[features_name]

    def remove_recordings(
        self,
    ):
        if self._recordings_path is not None:
            self._files_to_delete.append(self._recordings_path)

        self._recordings = None
        self._recordings_path = None

    def remove_classes(self, classes_name: str):
        if self._classes_paths[classes_name] is not None:
            self._files_to_delete.append(self._class_paths[classes_name])

        del self._classes[classes_name]
        del self._classes_paths[classes_name]

    def remove_enrollments(
        self,
        enrollments_name: str,
    ):
        if self._enrollments_paths[enrollments_name] is not None:
            self._files_to_delete.append(self._enrollments_paths[enrollments_name])

        del self._enrollments[enrollments_name]
        del self._enrollments_paths[enrollments_name]

    def remove_trials(
        self,
        trials_name: str,
    ):
        if self._trials_paths[trials_name] is not None:
            self._files_to_delete.append(self._trials_paths[trials_name])

        del self._trials[trials_name]
        del self._trials_paths[trials_name]

    def add_cols_to_segments(
        self,
        right_table: Union[InfoTable, pd.DataFrame, PathLike],
        column_names: Union[None, str, List[str], np.ndarray] = None,
        on: Union[str, List[str], np.ndarray] = "id",
        right_on: Union[None, str, List[str], np.ndarray] = None,
        remove_missing: bool = False,
        create_class_info: bool = False,
    ):
        if isinstance(right_table, (str, Path)):
            file_path = Path(right_table)
            if file_path.is_file():
                right_table = InfoTable.load(file_path)
            else:
                if right_table == "recordings":
                    right_table = self.recordings()
                elif right_table in self.features_keys():
                    right_table = self.features_value(right_table)
                elif right_table in self.classes_keys():
                    right_table = self.classes_value(right_table)
                else:
                    raise ValueError("%s not found", right_table)

        segments = self.segments(keep_loaded=True)
        num_segs_0 = len(segments)
        segments.add_columns(
            right_table,
            column_names,
            on=on,
            right_on=right_on,
            remove_missing=remove_missing,
        )
        if remove_missing and len(segments) < num_segs_0:
            self.clean()

        if create_class_info and column_names is not None:
            self.create_class_info_from_col(column_names)

    def create_class_info_from_col(
        self,
        column_names: Union[str, List[str], np.ndarray],
    ):
        if isinstance(column_names, str):
            column_names = [column_names]

        for col in column_names:
            if col not in self._classes:
                df = pd.DataFrame(
                    {"id": np.unique(self.segments(keep_loaded=True)[col])}
                )
                class_info = ClassInfo(df)
                self.add_classes(col, class_info)

    def clean(self, rebuild_class_idx=False):

        rec_ids = self.segments().recordings()
        self._recordings = self.recordings().filter(lambda df: df["id"].isin(rec_ids))

        ids = self.segments()["id"].values
        for k, table in self.features():
            self._features[k] = table.filter(lambda df: df["id"].isin(ids))

        for k, table in self.classes():
            class_ids = self.segments()[k].unique()
            self._classes[k] = table.filter(lambda df: df["id"].isin(class_ids))

        remove_keys = []
        for k, table in self.enrollments():
            table = table.filter(lambda df: df["segmentid"].isin(ids))
            if len(table) > 0:
                self._enrollments[k] = table
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
        rng = np.random.default_rng(seed=seed)

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
          HypDataset used for trials with trial list.
          HypDataset used for cohort.
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

    def remove_short_segments(self, min_length: float, length_name: str = "duration"):
        segments = self.segments()
        self._segments = segments.filter(lambda df: df[length_name] >= min_length)
        self.clean()

    def remove_classes_few_segments(
        self,
        class_name: str,
        min_segs: int,
        rebuild_idx: bool = False,
    ):
        segments = self.segments()
        classes, counts = np.unique(segments[class_name], return_counts=True)
        keep_classes = classes[counts >= min_segs]
        self._segments = segments.filter(lambda df: df[class_name].isin(keep_classes))
        self.clean()
        if rebuild_idx:
            class_info = self.classes_value(class_name)
            class_info.add_class_idx()

    def remove_classes_few_toomany_segments(
        self,
        class_name: str,
        min_segs: int,
        max_segs: int,
        rebuild_idx: bool = False,
    ):
        segments = self.segments()
        classes, counts = np.unique(segments[class_name], return_counts=True)
        if max_segs is None:
            keep_classes = classes[counts >= min_segs]
        else:
            keep_classes = classes[
                np.logical_and(counts >= min_segs, counts <= max_segs)
            ]
        self._segments = segments.filter(lambda df: df[class_name].isin(keep_classes))
        self.clean()
        if rebuild_idx:
            class_info = self.classes_value(class_name)
            class_info.add_class_idx()

    def rebuild_class_idx(self, class_name: str):
        class_info = self.classes_value(class_name)
        class_info.add_class_idx()

    def _segments_split(self, val_prob: float, rng: np.random.Generator):
        segments = self.segments()
        p = rng.permutation(len(segments))
        num_train = int(round((1 - val_prob) * len(p)))

        train_idx = p[:num_train]
        train_segs = segments.filter(iindex=train_idx)
        train_segs.sort()

        val_idx = p[num_train:]
        val_segs = segments.filter(iindex=val_idx)
        val_segs.sort()

        return train_segs, val_segs

    def _segments_split_joint_classes(
        self,
        val_prob: float,
        joint_classes: List[str],
        min_train_samples: int,
        rng: np.random.Generator,
    ):
        segments = self.segments()
        classes = segments[joint_classes].apply("-".join, axis=1)
        u_classes, class_ids = np.unique(classes, return_inverse=True)
        train_mask = np.zeros(len(segments), dtype=bool)
        kk = 0
        for c_id in range(len(u_classes)):
            idx = (class_ids == c_id).nonzero()[0]
            count = len(idx)
            p = rng.permutation(count)
            num_train = max(
                int(round((1 - val_prob) * count)), min(min_train_samples, count)
            )
            kk += count - num_train
            train_idx = idx[p[:num_train]]
            train_mask[train_idx] = True

        train_idx = train_mask.nonzero()[0]
        train_segs = segments.filter(iindex=train_idx)
        train_segs.sort()

        val_segs = segments.filter(iindex=train_idx, keep=False)
        val_segs.sort()

        return train_segs, val_segs

    def _segments_split_disjoint_classes(
        self,
        val_prob: float,
        disjoint_classes: List[str],
        rng: np.random.Generator,
    ):
        segments = self.segments()
        classes = segments[disjoint_classes].apply("-".join, axis=1)
        u_classes, class_ids = np.unique(classes, return_inverse=True)
        p = rng.permutation(len(u_classes))
        class_ids = p[class_ids]
        num_train = int(round((1 - val_prob) * len(segments)))
        train_mask = np.zeros(len(segments), dtype=bool)
        count_acc = 0
        for c_id in range(len(u_classes)):
            idx = (class_ids == c_id).nonzero()[0]
            train_mask[idx] = True
            count = len(idx)
            count_acc += count
            if count_acc >= num_train:
                break

        train_idx = train_mask.nonzero()[0]
        train_segs = segments.filter(iindex=train_idx)
        train_segs.sort()

        val_segs = segments.filter(iindex=train_idx, keep=False)
        val_segs.sort()

        return train_segs, val_segs

    def _segments_split_joint_and_disjoint_classes(
        self,
        val_prob: float,
        joint_classes: List[str],
        disjoint_clases: List[str],
        min_train_samples: int,
        rng: np.random.Generator,
    ):
        raise NotImplementedError("I'll implement this when I need it")
        segments = self.segments()
        j_classes = segments[joint_classes].apply("-".join, axis=1)
        ju_classes, j_class_ids = np.unique(j_classes, return_inverse=True)
        d_classes = segments[disjoint_classes].apply("-".join, axis=1)
        du_classes, d_class_ids = np.unique(d_classes, return_inverse=True)
        d_p = rng.permutation(len(du_classes))
        d_class_ids = d_p[d_class_ids]
        d_sort_idx = np.argsort(d_class_ids)
        d_sort_j_class_ids = j_class_ids[d_sort_idx]

        train_d_classes = set()
        for c_id in range(len(ju_classes)):
            idx = (j_sort_class_ids == c_id).nonzero()[0]
            count = len(idx)
            num_train = max(
                int(round((1 - val_prob) * count)), min(min_train_samples, count)
            )
            sel_d_class_ids = set(d_sort_idx[:num_train])
            train_d_classes = train_d_classes.union(sel_d_class_ids)

        train_mask = np.zeros(len(segments), dtype=bool)
        for c_id in train_d_classes:
            mask = d_class_ids == c_id
            train_mask[mask] = True

        train_idx = train_mask.nonzero()[0]
        train_segs = segments.filter(iindex=train_idx)
        train_segs.sort()

        val_segs = segments.filter(iindex=train_idx, keep=False)
        val_segs.sort()

        return train_segs, val_segs

    def split_train_val(
        self,
        val_prob: float,
        joint_classes: Optional[List[str]] = None,
        disjoint_classes: Optional[List[str]] = None,
        min_train_samples: int = 1,
        seed: int = 11235813,
    ):
        rng = np.random.default_rng(seed)
        if joint_classes is None and disjoint_classes is None:
            train_segs, val_segs = self._segments_split(val_prob, rng)
        elif joint_classes is not None and disjoint_classes is None:
            train_segs, val_segs = self._segments_split_joint_classes(
                val_prob,
                joint_classes,
                min_train_samples,
                rng,
            )
        elif joint_classes is None and disjoint_classes is not None:
            train_segs, val_segs = self._segments_split_disjoint_classes(
                val_prob,
                disjoint_classes,
                rng,
            )
        else:
            train_segs, val_segs = self._segments_split_joint_and_disjoint_classes(
                val_prob,
                joint_classes,
                disjoint_classes,
                min_train_samples,
                rng,
            )

        train_ds = self.clone()
        train_ds.set_segments(train_segs)
        train_ds.clean()

        val_ds = self.clone()
        val_ds.set_segments(val_segs)
        val_ds.clean()

        return train_ds, val_ds

    @classmethod
    def merge(cls, datasets):
        segments = []
        for dset in datasets:
            segs_dset = dset.segments(keep_loaded=False)
            if segs_dset is not None:
                segments.append(segs_dset)

        segments = SegmentSet.cat(segments)
        dataset = cls(segments)

        classes_keys = []
        for dset in datasets:
            classes_dset = list(dset.classes_keys())
            classes_keys.extend(classes_dset)

        classes_keys = list(set(classes_keys))
        for key in classes_keys:
            classes = []
            for dset in datasets:
                if key in dset.classes_keys():
                    classes_key = dset.classes_value(key, keep_loaded=False)
                    classes.append(classes_key)

            classes = ClassInfo.cat(classes)
            dataset.add_classes(classes_name=key, classes=classes)

        recordings = []
        for dset in datasets:
            recs_i = dset.recordings(keep_loaded=False)
            if recs_i is not None:
                recordings.append(recs_i)

        if recordings:
            recordings = RecordingSet.cat(recordings)
            dataset.set_recordings(recordings)

        features_keys = []
        for dset in datasets:
            features_dset = list(dset.features_keys())
            features_keys.extend(features_dset)

        features_keys = list(set(features_keys))
        for key in features_keys:
            features = []
            for dset in datasets:
                if key in dset.features_keys():
                    features_key = dset.features_value(key, keep_loaded=False)
                    features.append(features_key)

            features = FeatureSet.cat(features)
            dataset.add_features(features_name=key, features=features)

        # TODO: merge enrollments and trials
        # Usually you don't need that
        return dataset

    @classmethod
    def from_lhotse(
        cls,
        cuts: Optional[Union[lhotse.CutSet, PathLike]] = None,
        recordings: Optional[Union[lhotse.RecordingSet, PathLike]] = None,
        supervisions: Optional[Union[lhotse.SupervisionSet, PathLike]] = None,
    ):
        """Creates a Hyperion Dataset from a lhotse CutSet or
        from a lhotse RecordingSet + SupervisionSet

        Args:
          cuts: lhotse CutSet manifest or file
          recordings: lhotse RecordingSet manifest or file
          supervisions: lhotse SupervisionSet manifest or file.

        Returns
          HypDataset object
        """
        assert cuts is not None or supervisions is not None
        if cuts is not None:
            if isinstance(cuts, (str, Path)):
                cuts = lhotse.CutSet.from_file(cuts)
        else:
            if isinstance(supervisions, (str, Path)):
                supervisions = lhotse.SupervisionSet.from_file(supervisions)

            if recordings is not None and isinstance(recordings, (str, Path)):
                recordings = lhotse.RecordingSet.from_file(recordings)

            cuts = lhotse.CutSet.from_manifests(
                recordings=recordings, supervisions=supervisions
            )

        from lhotse import MonoCut, Recording, SupervisionSegment

        supervision_keys = [
            "speaker",
            "gender",
            "language",
            "emotion",
            "text",
            "duration",
        ]
        recs_df = []
        segs_df = []
        for cut in cuts:
            supervision = cut.supervisions[0]
            recording = cut.recording
            seg_dict = {"id": cut.id}
            recording = cut.recording
            if recording is not None:
                # if recording.id != cut.id:
                #     seg_dict["recording_id"] = recording.id

                rec_dict = {
                    "id": cut.id,
                    "sampling_rate": recording.sampling_rate,
                    "duration": recording.duration,
                }
                source = recording.sources[0]
                assert len(recording.sources) == 1
                assert source.type in ["file", "command"]
                rec_dict["storage_path"] = source.source
                assert recording.transforms is None, f"{recording.transforms}"
                recs_df.append(rec_dict)

            for key in supervision_keys:
                if hasattr(supervision, key):
                    val = getattr(supervision, key)
                    if val is not None:
                        seg_dict[key] = val

            if supervision.custom is not None:
                for key, val in supervision.custom:
                    if val is not None:
                        seg_dict[key] = val

            segs_df.append(seg_dict)

        recs_df = pd.DataFrame(recs_df)
        segs_df = pd.DataFrame(segs_df)
        recordings = RecordingSet(recs_df)
        segments = SegmentSet(segs_df)
        class_names = ["speaker", "language", "emotion", "gender"]
        classes = {}
        for key in class_names:
            if key in segments:
                uniq_classes = np.unique(segments[key])
                classes[key] = ClassInfo(pd.DataFrame({"id": uniq_classes}))

        if not classes:
            classes = None

        dataset = cls(segments=segments, classes=classes, recordings=recordings)
        return dataset

    @classmethod
    def from_kaldi(
        cls,
        kaldi_data_dir: PathLike,
    ):
        """Creates a Hyperion Dataset from a Kaldi data dir

        Args:
          kaldi_data_dir: Kaldi data directory

        Returns
          HypDataset object
        """
        kaldi_data_dir = Path(kaldi_data_dir)

        kaldi_files = ["utt2lang", "utt2dur", "utt2text"]
        attributes = ["language", "duration", "text"]

        k_file = kaldi_data_dir / "utt2spk"
        from .utt2info import Utt2Info

        utt2spk = Utt2Info.load(k_file)
        df_segs = pd.DataFrame({"id": utt2spk.key, "speaker": utt2spk.info})
        segments = SegmentSet(df_segs)
        del utt2spk

        for att, k_file in zip(kaldi_files, attributes):
            k_file = kaldi_data_dir / k_file
            if k_file.is_file():
                u2i = Utt2Info.load(k_file)
                segments.loc[u2i.key, att] = u2i.info

        k_file = kaldi_data_dir / "spk2gender"
        if k_file.is_file():
            segments["gender"] = "N/A"
            s2g = Utt2Info.load(k_file)
            for spk in s2g.key:
                g = s2g[spk]
                segments.loc[segments["speaker"] == spk, "gender"] = g

        kaldi_files = ["feats.scp", "vad.scp"]
        attributes = ["feats", "vad"]
        features = None
        from .scp_list import SCPList

        for att, k_file in zip(kaldi_files, attributes):
            k_file = kaldi_data_dir / k_file
            if k_file.is_file():
                scp = SCPList.load(k_file)
                feats_dict = {"id": scp.key, "storage_path": scp.file_path}
                if scp.offset is not None:
                    feats_dict["storage_byte"] = scp.offset
                df_feats = pd.DataFrame(feats_dict)
                if features is None:
                    features = {}
                features["att"] = FeatureSet(df_feats)

        recordings = None
        k_file = kaldi_data_dir / "wav.scp"
        if k_file.is_file():
            scp = SCPList.load(k_file)
            wav_dict = {"id": scp.key, "storage_path": scp.file_path}
            df_recs = pd.DataFrame(wav_dict)
            recordings = RecordingSet(df_recs)
            recordings.get_durations()
            if "duration" not in segments:
                segments["duration"] = recordings.loc[segments["id"], "duration"]

        class_names = ["speaker", "language", "emotion", "gender"]
        classes = {}
        for key in class_names:
            if key in segments:
                uniq_classes = np.unique(segments[key])
                classes[key] = ClassInfo(pd.DataFrame({"id": uniq_classes}))

        if not classes:
            classes = None

        dataset = cls(
            segments=segments, classes=classes, recordings=recordings, features=features
        )
        return dataset
