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
from .diarization_set import DiarizationSet
from .enrollment_map import EnrollmentMap
from .feature_set import FeatureSet
from .image_set import ImageSet
from .info_table import InfoTable
from .misc import PathLike
from .recording_set import RecordingSet
from .segment_set import SegmentSet
from .sparse_trial_key import SparseTrialKey
from .trial_key import TrialKey
from .trial_ndx import TrialNdx
from .vad_set import VADSet
from .video_set import VideoSet


class HyperDataset:
    """Container that groups segments with their related resources.

    The dataset keeps references to tables (segments, recordings, features, etc.)
    either as in-memory objects or filesystem paths. Paths are loaded lazily
    when an accessor is called so large datasets can be defined without reading
    everything into memory upfront.

    Attributes:
      segments: SegmentSet object or path; this is the only required table.
      classes: Mapping from class name to ClassInfo object or path.
      recordings: RecordingSet object or path aligned with segment `recording`.
      images: ImageSet object or path aligned with segment `image`.
      videos: VideoSet object or path aligned with segment `video`.
      features: Mapping from feature name to FeatureSet object or path.
      vads: Mapping from VAD name to VADSet object or path.
      diarizations: Mapping from diarization name to DiarizationSet object or path.
      enrollments: Mapping from enrollment name to EnrollmentMap object or path.
      trials: Mapping from trial name to TrialKey/TrialNdx/SparseTrialKey object or path.
      sparse_trials: If True, load trials using SparseTrialKey to save memory.
      table_sep: Default column separator when reading/writing tables.
      trials_sep: Separator for trial manifests; falls back to ``table_sep`` when None.
    """

    def __init__(
        self,
        segments: Union[SegmentSet, PathLike],
        classes: Optional[Dict[str, Union[ClassInfo, PathLike]]] = None,
        recordings: Optional[Union[RecordingSet, PathLike]] = None,
        images: Optional[Union[ImageSet, PathLike]] = None,
        videos: Optional[Union[VideoSet, PathLike]] = None,
        features: Optional[Dict[str, Union[FeatureSet, PathLike]]] = None,
        vads: Optional[Dict[str, Union[VADSet, PathLike]]] = None,
        diarizations: Optional[Dict[str, Union[DiarizationSet, PathLike]]] = None,
        enrollments: Optional[Dict[str, Union[EnrollmentMap, PathLike]]] = None,
        trials: Optional[
            Dict[str, Union[TrialKey, TrialNdx, SparseTrialKey, PathLike]]
        ] = None,
        sparse_trials: bool = False,
        table_sep: Optional[str] = None,
        trials_sep: Optional[str] = None,
    ):
        """Initialize the dataset wrapper and optionally register auxiliary tables.

        Args:
            segments: SegmentSet instance or path to the segments table; required anchor for the dataset.
            classes: Optional mapping from class name to ClassInfo object or path containing class labels.
            recordings: Optional RecordingSet object or path aligned with the ``recording`` column.
            images: Optional ImageSet object or path aligned with the ``image`` column.
            videos: Optional VideoSet object or path aligned with the ``video`` column.
            features: Optional mapping from feature name to FeatureSet object or path.
            vads: Optional mapping from VAD name to VADSet object or path.
            diarizations: Optional mapping from diarization name to DiarizationSet object or path.
            enrollments: Optional mapping from enrollment name to EnrollmentMap object or path.
            trials: Optional mapping from trial name to TrialKey/TrialNdx/SparseTrialKey object or path.
            sparse_trials: If True, load trial files using SparseTrialKey for memory efficiency.
            table_sep: Column separator used when reading or writing tabular manifests.
            trials_sep: Optional separator for trial manifests; defaults to ``table_sep`` when None.
        """
        if isinstance(segments, SegmentSet):
            self._segments = segments
            self._segments_path = None
        else:
            assert isinstance(segments, (str, Path))
            self._segments = None
            self._segments_path = Path(segments)

        self._classes, self._classes_paths = self._parse_dict_args(classes, ClassInfo)
        self._recordings = None
        self._recordings_path = None
        if recordings is not None:
            if isinstance(recordings, RecordingSet):
                self._recordings = recordings
            else:
                assert isinstance(recordings, (str, Path))
                self._recordings_path = Path(recordings)

        self._images = None
        self._images_path = None
        if images is not None:
            if isinstance(images, ImageSet):
                self._images = images
            else:
                assert isinstance(images, (str, Path))
                self._images_path = Path(images)

        self._videos = None
        self._videos_path = None
        if videos is not None:
            if isinstance(videos, VideoSet):
                self._videos = videos
            else:
                assert isinstance(videos, (str, Path))
                self._videos_path = Path(videos)

        self._features, self._features_paths = self._parse_dict_args(
            features, FeatureSet
        )
        self._vads, self._vads_paths = self._parse_dict_args(vads, VADSet)
        self._diarizations, self._diarizations_paths = self._parse_dict_args(
            diarizations, DiarizationSet
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
        self.trials_sep = trials_sep if trials_sep is not None else table_sep
        self._files_to_delete = []
        self.fix_segments_dtypes()

    def fix_segments_dtypes(self):
        """Ensure any class columns in the segments table are stored as strings."""
        if self._segments is not None:
            self._fix_segments_dtypes(self._segments)

    def _fix_segments_dtypes(self, segments):
        """Convert class columns in a SegmentSet to string dtype.

        Args:
            segments: SegmentSet whose columns will be adjusted in-place.
        """
        # ids in class_infos should be strings in segment set columns
        for k in self.classes_keys():
            if k in segments:
                segments.convert_col_to_str(k)

    def describe(self):
        """Summarize dataset counts and duration, logging a human-readable message.

        Returns:
            Dict[str, Union[int, float, str]]: Counts per component plus a ``msg`` field.
        """
        segments = self.segments(keep_loaded=False)
        info = {"num_segments": len(segments)}
        for class_name, class_info in self.classes(keep_loaded=False):
            info[f"num_{class_name}s"] = len(class_info)
        if "duration" in segments:
            info["duration_hours"] = segments["duration"].sum() / 3600

        msg = "Dataset contains %s" % (", ".join([f"{k}={v}" for k, v in info.items()]))
        info["msg"] = msg
        logging.info(msg)
        return info

    def get_dataset_files(self):
        """Collect all manifest file paths referenced by the dataset.

        Returns:
            List[Path]: Paths for segments, recordings/videos, and auxiliary manifests.
        """
        file_paths = []
        for file_path in [
            self._segments_path,
            self._recordings_path,
            self._videos_path,
        ]:
            if file_path is not None:
                file_paths.append(file_path)

        for path_dict in [
            self._features_paths,
            self._vads_paths,
            self._diarizations_paths,
            self._enrollments_paths,
            self._trials_paths,
        ]:
            if path_dict is None:
                continue
            for k, v in path_dict.items():
                file_paths.append(v)

        return file_paths

    def _delete_files(self, dataset_dir):
        """Delete files queued for removal if they are not part of the saved dataset.

        Args:
            dataset_dir: Target directory where dataset manifests are stored.
        """
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
        """Split a mapping into separate object and path dictionaries.

        Args:
            data: Mapping whose values are either instances of ``types`` or paths.
            types: Class or tuple of classes expected for in-memory objects.

        Returns:
            Tuple[Optional[Dict[str, object]], Optional[Dict[str, Path]]]: Objects and paths keyed by name.
        """
        if data is None:
            return None, None

        assert isinstance(data, dict)
        objects = {k: (v if isinstance(v, types) else None) for k, v in data.items()}
        paths = {
            k: (v if isinstance(v, (str, Path)) else None) for k, v in data.items()
        }
        return objects, paths

    def clone(self):
        """Return a deep copy of the dataset."""
        return deepcopy(self)

    def segments(self, keep_loaded: bool = True):
        """Access the segments table, loading from disk if needed.

        Args:
            keep_loaded: If True, cache the loaded SegmentSet on the instance.

        Returns:
            SegmentSet: Segment metadata for the dataset.
        """
        if self._segments is None:
            assert self._segments_path is not None
            segments = SegmentSet.load(self._segments_path, sep=self.table_sep)
            self._fix_segments_dtypes(segments)
            if keep_loaded:
                self._segments = segments
            return segments

        return self._segments

    def __len__(self):
        """Number of segments in the dataset."""
        return len(self.segments())

    @property
    def has_recordings(self):
        """Whether a recordings manifest is available (loaded or path)."""
        return self._recordings is not None or self._recordings_path is not None

    @property
    def has_images(self):
        """Whether an images manifest is available (loaded or path)."""
        return self._images is not None or self._images_path is not None

    @property
    def has_videos(self):
        """Whether a videos manifest is available (loaded or path)."""
        return self._videos is not None or self._videos_path is not None

    def recordings(self, keep_loaded: bool = True):
        """Access the recordings table, loading from disk if needed.

        Args:
            keep_loaded: If True, cache the loaded RecordingSet.

        Returns:
            RecordingSet: Recording metadata aligned with segments.
        """
        if self._recordings is None:
            assert self._recordings_path is not None
            recordings = RecordingSet.load(self._recordings_path, sep=self.table_sep)
            if keep_loaded:
                self._recordings = recordings
            return recordings

        return self._recordings

    def images(self, keep_loaded: bool = True):
        """Access the images table, loading from disk if needed.

        Args:
            keep_loaded: If True, cache the loaded ImageSet.

        Returns:
            ImageSet: Image metadata aligned with segments.
        """
        if self._images is None:
            assert self._images_path is not None
            images = ImageSet.load(self._images_path)
            if keep_loaded:
                self._images = images
            return images

        return self._images

    def videos(self, keep_loaded: bool = True):
        """Access the videos table, loading from disk if needed.

        Args:
            keep_loaded: If True, cache the loaded VideoSet.

        Returns:
            VideoSet: Video metadata aligned with segments.
        """
        if self._videos is None:
            assert self._videos_path is not None
            videos = VideoSet.load(self._videos_path)
            if keep_loaded:
                self._videos = videos
            return videos

        return self._videos

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
        """Return names of feature sets present in the dataset."""
        if self._features is not None:
            return self._features.keys()
        elif self._features_paths is not None:
            return self._features_paths.keys()
        else:
            return {}

    def features_value(self, key: str, keep_loaded: bool = True):
        """Access a feature manifest by name.

        Args:
            key: Name of the feature set.
            keep_loaded: If True, cache the loaded FeatureSet.

        Returns:
            FeatureSet: Feature manifest referenced by ``key``.
        """
        if self._features[key] is None:
            assert self._features_paths[key] is not None
            features = FeatureSet.load(self._features_paths[key], sep=self.table_sep)
            if keep_loaded:
                self._features[key] = features
            return features

        return self._features[key]

    def vads_keys(self):
        """Return names of VAD sets present in the dataset."""
        if self._vads is not None:
            return self._vads.keys()
        elif self._vads_paths is not None:
            return self._vads_paths.keys()
        else:
            return {}

    def vads_value(self, key: str, keep_loaded: bool = True):
        """Access a VAD manifest by name.

        Args:
            key: Name of the VAD set.
            keep_loaded: If True, cache the loaded VADSet.

        Returns:
            VADSet: VAD manifest referenced by ``key``.
        """
        if self._vads[key] is None:
            assert self._vads_paths[key] is not None
            vads = VADSet.load(self._vads_paths[key], sep=self.table_sep)
            if keep_loaded:
                self._vads[key] = vads
            return vads

        return self._vads[key]

    def diarizations_keys(self):
        """Return names of diarization sets present in the dataset."""
        if self._diarizations is not None:
            return self._diarizations.keys()
        elif self._diarizations_paths is not None:
            return self._diarizations_paths.keys()
        else:
            return {}

    def diarizations_value(self, key: str, keep_loaded: bool = True):
        """Access a diarization manifest by name.

        Args:
            key: Name of the diarization set.
            keep_loaded: If True, cache the loaded DiarizationSet.

        Returns:
            DiarizationSet: Diarization manifest referenced by ``key``.
        """
        if self._diarizations[key] is None:
            assert self._diarizations_paths[key] is not None
            diarizations = DiarizationSet.load(
                self._diarizations_paths[key], sep=self.table_sep
            )
            if keep_loaded:
                self._diarizations[key] = diarizations
            return diarizations

        return self._vads[key]

    def classes_keys(self):
        """Return names of class info tables present in the dataset."""
        if self._classes is not None:
            return self._classes.keys()
        elif self._classes_paths is not None:
            return self._classes_paths.keys()
        else:
            return {}

    def classes_value(self, key: str, keep_loaded: bool = True):
        """Access a class info manifest by name.

        Args:
            key: Name of the class info table.
            keep_loaded: If True, cache the loaded ClassInfo.

        Returns:
            ClassInfo: Class metadata referenced by ``key``.
        """
        if self._classes[key] is None:
            assert self._classes_paths[key] is not None
            classes = ClassInfo.load(self._classes_paths[key], self.table_sep)
            if keep_loaded:
                self._classes[key] = classes
            return classes

        return self._classes[key]

    def enrollments_value(self, key: str, keep_loaded: bool = True):
        """Access an enrollment map by name.

        Args:
            key: Name of the enrollment map.
            keep_loaded: If True, cache the loaded EnrollmentMap.

        Returns:
            EnrollmentMap: Enrollment manifest referenced by ``key``.
        """
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
        """Access a trials object by name, loading lazily from disk.

        Args:
            key: Name of the trials entry.
            keep_loaded: If True, cache the loaded trials structure.

        Returns:
            Union[TrialKey, TrialNdx, SparseTrialKey]: Trials data referenced by ``key``.
        """
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
        """Iterate over all feature sets, loading lazily if necessary.

        Args:
            keep_loaded: If True, cache each loaded FeatureSet.

        Yields:
            Tuple[str, FeatureSet]: Feature name and manifest.
        """
        if self._features is None:
            yield from ()
        else:
            for key in self._features.keys():
                yield key, self.features_value(key, keep_loaded)

    def vads(self, keep_loaded: bool = True):
        """Iterate over all VAD sets, loading lazily if necessary.

        Args:
            keep_loaded: If True, cache each loaded VADSet.

        Yields:
            Tuple[str, VADSet]: VAD name and manifest.
        """
        if self._vads is None:
            yield from ()
        else:
            for key in self._vads.keys():
                yield key, self.vads_value(key, keep_loaded)

    def diarizations(self, keep_loaded: bool = True):
        """Iterate over all diarization sets, loading lazily if necessary.

        Args:
            keep_loaded: If True, cache each loaded DiarizationSet.

        Yields:
            Tuple[str, DiarizationSet]: Diarization name and manifest.
        """
        if self._diarizations is None:
            yield from ()
        else:
            for key in self._diarizations.keys():
                yield key, self.diarizations_value(key, keep_loaded)

    def classes(self, keep_loaded: bool = True):
        """Iterate over all class info tables, loading lazily if necessary.

        Args:
            keep_loaded: If True, cache each loaded ClassInfo.

        Yields:
            Tuple[str, ClassInfo]: Class name and table.
        """
        if self._classes is None:
            yield from ()
        else:
            for key in self._classes.keys():
                yield key, self.classes_value(key, keep_loaded)

    def enrollments(self, keep_loaded: bool = True):
        """Iterate over all enrollment maps, loading lazily if necessary.

        Args:
            keep_loaded: If True, cache each loaded EnrollmentMap.

        Yields:
            Tuple[str, EnrollmentMap]: Enrollment name and map.
        """
        if self._enrollments is None:
            yield from ()
        else:
            for key in self._enrollments.keys():
                yield key, self.enrollments_value(key, keep_loaded)

    def trials(self, keep_loaded: bool = True):
        """Iterate over all trials, loading lazily if necessary.

        Args:
            keep_loaded: If True, cache each loaded trials object.

        Yields:
            Tuple[str, Union[TrialKey, TrialNdx, SparseTrialKey]]: Trial name and data.
        """
        if self._trials is None:
            yield from ()
        else:
            for key in self._trials.keys():
                yield key, self.trials_value(key, keep_loaded)

    @staticmethod
    def resolve_dataset_path(dataset_path):
        """Normalize a dataset path to directory and YAML manifest.

        Args:
            dataset_path: Path to a dataset directory or dataset YAML file.

        Returns:
            Tuple[Path, Path]: Dataset directory and dataset YAML file path.
        """
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
        """Resolve a manifest path relative to the dataset directory.

        Args:
            dataset_dir: Base directory for the dataset.
            file_path: Absolute or relative manifest path.

        Returns:
            Path: Resolved file path.
        """
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
        """Persist the dataset manifests to disk.

        Args:
            dataset_path: Directory to hold manifests or path to a dataset YAML file.
            update_paths: Whether to update internal file paths after saving.
            table_sep: Separator to use when writing tabular files (overrides instance default).
            force_save_all: If True, save every table; otherwise only save loaded/changed files.
                Trials use ``trials_sep`` when provided.

        Returns:
            None
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
        trials_sep: Optional[str] = None,
    ):
        """Save only manifests that changed or are missing in the target directory.

        Args:
            dataset_path: Directory to hold manifests or path to a dataset YAML file.
            update_paths: Whether to update internal file paths after saving.
            table_sep: Separator to use when writing tabular files (overrides instance default).
            trials_sep: Separator to use when writing trial files (overrides instance default).

        Returns:
            None
        """
        table_sep = self.table_sep if table_sep is None else table_sep
        trials_sep = self.trials_sep if trials_sep is None else trials_sep
        if update_paths:
            self.table_sep = table_sep
            self.trials_sep = trials_sep

        table_ext = ".tsv" if table_sep == "\t" else ".csv"
        trials_ext = ".tsv" if trials_sep == "\t" else ".csv"

        dataset_dir, dataset_file = HyperDataset.resolve_dataset_path(dataset_path)
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

        if self.has_recordings:
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

        if self.has_images:
            file_name = f"images{table_ext}"
            dataset["images"] = file_name
            file_path = dataset_dir / file_name
            if (
                self._images is not None
                or file_path != self._images_path
                or not file_path.exists()
            ):
                self.images(keep_loaded=False).save(file_path, sep=table_sep)
                if update_paths:
                    self._images_path = file_path

        if self.has_videos:
            file_name = f"videos{table_ext}"
            dataset["videos"] = file_name
            file_path = dataset_dir / file_name
            if (
                self._videos is not None
                or file_path != self._videos_path
                or not file_path.exists()
            ):
                self.videos(keep_loaded=False).save(file_path, sep=table_sep)
                if update_paths:
                    self._videos_path = file_path

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

        if self._vads is not None:
            file_names = {}
            for k in self._vads.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._vads[k] is not None
                    or file_path != self._vads_paths[k]
                    or not file_path.exists()
                ):
                    v = self.vads_value(k, keep_loaded=False)
                    v.save(file_path, sep=table_sep)
                    if update_paths:
                        self._vads_paths[k] = file_path

            if file_names:
                dataset["vads"] = file_names

        if self._diarizations is not None:
            file_names = {}
            for k in self._diarizations.keys():
                file_name = k + table_ext
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                if (
                    self._diarizations[k] is not None
                    or file_path != self._diarizations_paths[k]
                    or not file_path.exists()
                ):
                    v = self.diarizations_value(k, keep_loaded=False)
                    v.save(file_path, sep=table_sep)
                    if update_paths:
                        self._diarizations_paths[k] = file_path

            if file_names:
                dataset["diarizations"] = file_names

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
                file_name = k + trials_ext
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
        trials_sep: Optional[str] = None,
    ):
        """Save every manifest to disk, regardless of change tracking.

        Args:
            dataset_path: Directory to hold manifests or path to a dataset YAML file.
            update_paths: Whether to update internal file paths after saving.
            table_sep: Separator to use when writing tabular files (overrides instance default).
            trials_sep: Separator to use when writing trial files (overrides instance default).

        Returns:
            None
        """
        table_sep = self.table_sep if table_sep is None else table_sep
        trials_sep = self.trials_sep if trials_sep is None else trials_sep
        if update_paths:
            self.table_sep = table_sep
            self.trials_sep = trials_sep

        table_ext = ".tsv" if table_sep == "\t" else ".csv"
        trials_ext = ".tsv" if trials_sep == "\t" else ".csv"

        dataset_dir, dataset_file = HyperDataset.resolve_dataset_path(dataset_path)
        dataset = {}
        file_name = f"segments{table_ext}"
        dataset["segments"] = file_name
        file_path = dataset_dir / file_name
        self.segments(keep_loaded=False).save(file_path, sep=table_sep)
        if update_paths:
            self._segments_path = file_path

        if self.has_recordings:
            file_name = f"recordings{table_ext}"
            dataset["recordings"] = file_name
            file_path = dataset_dir / file_name
            self.recordings(keep_loaded=False).save(file_path, sep=table_sep)
            if update_paths:
                self._recordings_path = file_path

        if self.has_images:
            file_name = f"images{table_ext}"
            dataset["images"] = file_name
            file_path = dataset_dir / file_name
            self.images(keep_loaded=False).save(file_path, sep=table_sep)
            if update_paths:
                self._images_path = file_path

        if self.has_videos:
            file_name = f"videos{table_ext}"
            dataset["videos"] = file_name
            file_path = dataset_dir / file_name
            self.videos(keep_loaded=False).save(file_path, sep=table_sep)
            if update_paths:
                self._videos_path = file_path

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
        for k, v in self.vads(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path, sep=table_sep)
            if update_paths:
                self._vads_paths[k] = file_path

        if file_names:
            dataset["vads"] = file_names

        file_names = {}
        for k, v in self.diarizations(keep_loaded=False):
            file_name = k + table_ext
            file_names[k] = file_name
            file_path = dataset_dir / file_name
            v.save(file_path, sep=table_sep)
            if update_paths:
                self._diarizations_paths[k] = file_path

        if file_names:
            dataset["diarizations"] = file_names

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
            file_name = k + trials_ext
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
        """Eagerly load every registered manifest into memory."""
        self.segments()
        self.recordings()
        self.images()
        self.videos()

        for k, v in self.features():
            pass

        for k, v in self.vads():
            pass

        for k, v in self.diarizations():
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
        """Instantiate a dataset from a manifest directory or YAML file.

        Args:
            dataset_path: Directory containing manifests or a dataset YAML file.
            lazy: If True, defer loading manifests until accessed.
            sparse_trials: If True, load trial files as SparseTrialKey.

        Returns:
            HyperDataset: Dataset pointing to the referenced manifests.
        """
        dataset_dir, dataset_file = HyperDataset.resolve_dataset_path(dataset_path)
        with open(dataset_file, "r") as f:
            dataset = yaml.safe_load(f)

        assert "segments" in dataset
        segments = HyperDataset.resolve_file_path(dataset_dir, dataset["segments"])
        classes = None
        recordings = None
        images = None
        videos = None
        features = None
        vads = None
        diarizations = None
        enrollments = None
        trials = None
        if "classes" in dataset:
            classes = {}
            for k, v in dataset["classes"].items():
                classes[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        if "recordings" in dataset:
            recordings = HyperDataset.resolve_file_path(
                dataset_dir, dataset["recordings"]
            )
            # recordings = {}
            # for k, v in dataset["recordings"].items():
            #     recordings[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        if "images" in dataset:
            images = HyperDataset.resolve_file_path(dataset_dir, dataset["images"])

        if "videos" in dataset:
            videos = HyperDataset.resolve_file_path(dataset_dir, dataset["videos"])

        if "features" in dataset:
            features = {}
            for k, v in dataset["features"].items():
                features[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        if "vads" in dataset:
            vads = {}
            for k, v in dataset["vads"].items():
                vads[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        if "diarizations" in dataset:
            diarizations = {}
            for k, v in dataset["diarizations"].items():
                diarizations[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        if "enrollments" in dataset:
            enrollments = {}
            for k, v in dataset["enrollments"].items():
                enrollments[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        if "trials" in dataset:
            trials = {}
            for k, v in dataset["trials"].items():
                trials[k] = HyperDataset.resolve_file_path(dataset_dir, v)

        dataset = cls(
            segments,
            classes,
            recordings,
            images,
            videos,
            features,
            vads,
            diarizations,
            enrollments,
            trials,
            sparse_trials=sparse_trials,
        )
        if not lazy:
            dataset.update_from_disk()

        return dataset

    def set_segments(
        self,
        segments: Union[PathLike, SegmentSet],
    ):
        """Replace the segments table reference.

        Args:
            segments: SegmentSet instance or path to a segments manifest.

        Returns:
            None
        """
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
        """Attach a recordings manifest to the dataset.

        Args:
            recordings: RecordingSet instance or path to a recordings manifest.
            update_seg_durs: If True, populate segment durations from recordings.

        Returns:
            None
        """
        if isinstance(recordings, (str, Path)):
            self._recordings = None
            self._recordings_path = Path(recordings)
        elif isinstance(recordings, RecordingSet):
            self._recordings = recordings
            self._recordings_path = None
        else:
            raise ValueError()

        if update_seg_durs:
            rec_ids = self.segments(keep_loaded=True).recording()
            self.segments()["duration"] = self.recordings().loc[rec_ids, "duration"]

    def set_images(
        self,
        images: Union[PathLike, ImageSet],
    ):
        """Attach an images manifest to the dataset.

        Args:
            images: ImageSet instance or path to an images manifest.

        Returns:
            None
        """
        if isinstance(images, (str, Path)):
            self._images = None
            self._images_path = Path(images)
        elif isinstance(images, ImageSet):
            self._images = images
            self._images_path = None
        else:
            raise ValueError()

    def set_videos(
        self,
        videos: Union[PathLike, VideoSet],
        update_seg_durs: bool = False,
    ):
        """Attach a videos manifest to the dataset.

        Args:
            videos: VideoSet instance or path to a videos manifest.
            update_seg_durs: If True, populate segment durations from videos.

        Returns:
            None
        """
        if isinstance(videos, (str, Path)):
            self._videos = None
            self._videos_path = Path(videos)
        elif isinstance(videos, VideoSet):
            self._videos = videos
            self._videos_path = None
        else:
            raise ValueError()

        if update_seg_durs:
            rec_ids = self.segments(keep_loaded=True).recording()
            self.segments()["duration"] = self.videos().loc[rec_ids, "duration"]

    def add_features(self, features_name: str, features: Union[PathLike, FeatureSet]):
        """Register a feature manifest under a given name.

        Args:
            features_name: Identifier for the feature set.
            features: FeatureSet instance or path to a features manifest.

        Returns:
            None
        """
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

    def add_vads(self, vads_name: str, vads: Union[PathLike, VADSet]):
        """Register a VAD manifest under a given name.

        Args:
            vads_name: Identifier for the VAD set.
            vads: VADSet instance or path to a VAD manifest.

        Returns:
            None
        """
        if self._vads is None:
            self._vads = {}
            self._vads_paths = {}

        if isinstance(vads, (str, Path)):
            self._vads[vads_name] = None
            self._vads_paths[vads_name] = vads
        elif isinstance(vads, VADSet):
            self._vads[vads_name] = vads
            self._vads_paths[vads_name] = None
        else:
            raise ValueError()

    def add_diarizations(
        self, diarizations_name: str, diarizations: Union[PathLike, DiarizationSet]
    ):
        """Register a diarization manifest under a given name.

        Args:
            diarizations_name: Identifier for the diarization set.
            diarizations: DiarizationSet instance or path to a diarization manifest.

        Returns:
            None
        """
        if self._diarizations is None:
            self._diarizations = {}
            self._diarizations_paths = {}

        if isinstance(diarizations, (str, Path)):
            self._diarizations[diarizations_name] = None
            self._diarizations_paths[diarizations_name] = diarizations
        elif isinstance(diarizations, DiarizationSet):
            self._diarizations[diarizations_name] = diarizations
            self._diarizations_paths[diarizations_name] = None
        else:
            raise ValueError()

    def add_classes(self, classes_name: str, classes: Union[PathLike, ClassInfo]):
        """Register a class info table under a given name.

        Args:
            classes_name: Identifier for the class table.
            classes: ClassInfo instance or path to a class manifest.

        Returns:
            None
        """
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
        """Register an enrollment map under a given name.

        Args:
            enrollments_name: Identifier for the enrollment map.
            enrollments: EnrollmentMap instance or path to an enrollment manifest.

        Returns:
            None
        """
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
        """Register a trials object under a given name.

        Args:
            trials_name: Identifier for the trials entry.
            trials: TrialKey, TrialNdx, SparseTrialKey instance or path to a trials file.

        Returns:
            None
        """
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

    def remove_recordings(
        self,
    ):
        """Detach recordings and mark backing file for deletion if present."""
        if self._recordings_path is not None:
            self._files_to_delete.append(self._recordings_path)

        self._recordings = None
        self._recordings_path = None

    def remove_images(
        self,
    ):
        """Detach images and mark backing file for deletion if present."""
        if self._images_path is not None:
            self._files_to_delete.append(self._images_path)

        self._images = None
        self._images_path = None

    def remove_videos(
        self,
    ):
        """Detach videos and mark backing file for deletion if present."""
        if self._videos_path is not None:
            self._files_to_delete.append(self._videos_path)

        self._videos = None
        self._videos_path = None

    def remove_features(self, features_name: str):
        """Remove a feature set and optionally delete its manifest file.

        Args:
            features_name: Identifier of the feature set to remove.

        Returns:
            None
        """
        if self._features_paths is None or features_name not in self._features_paths:
            logging.warning("Features %s not found in dataset", features_name)
            return

        if self._features_paths[features_name] is not None:
            self._files_to_delete.append(self._features_paths[features_name])

        del self._features[features_name]
        del self._features_paths[features_name]

    def remove_vads(self, vads_name: str):
        """Remove a VAD set and optionally delete its manifest file.

        Args:
            vads_name: Identifier of the VAD set to remove.

        Returns:
            None
        """
        if self._vads_paths is None or vads_name not in self._vads_paths:
            logging.warning("VAD %s not found in dataset", vads_name)
            return

        if self._vads_paths[vads_name] is not None:
            self._files_to_delete.append(self._vads_paths[vads_name])

        del self._vads[vads_name]
        del self._vads_paths[vads_name]

    def remove_diarizations(self, diarizations_name: str):
        """Remove a diarization set and optionally delete its manifest file.

        Args:
            diarizations_name: Identifier of the diarization set to remove.

        Returns:
            None
        """
        if (
            self._diarizations_paths is None
            or diarizations_name not in self._diarizations_paths
        ):
            logging.warning("Diarization %s not found in dataset", diarizations_name)
            return

        if self._diarizations_paths[diarizations_name] is not None:
            self._files_to_delete.append(self._diarizations_paths[diarizations_name])

        del self._diarizations[diarizations_name]
        del self._diarizations_paths[diarizations_name]

    def remove_classes(self, classes_name: str):
        """Remove a class info table and optionally delete its manifest file.

        Args:
            classes_name: Identifier of the class info table to remove.

        Returns:
            None
        """
        if self._classes_paths[classes_name] is not None:
            self._files_to_delete.append(self._classes_paths[classes_name])

        del self._classes[classes_name]
        del self._classes_paths[classes_name]

    def remove_enrollments(
        self,
        enrollments_name: str,
    ):
        """Remove an enrollment map and optionally delete its manifest file.

        Args:
            enrollments_name: Identifier of the enrollment map to remove.

        Returns:
            None
        """
        if self._enrollments_paths[enrollments_name] is not None:
            self._files_to_delete.append(self._enrollments_paths[enrollments_name])

        del self._enrollments[enrollments_name]
        del self._enrollments_paths[enrollments_name]

    def remove_trials(
        self,
        trials_name: str,
    ):
        """Remove a trials entry and optionally delete its manifest file.

        Args:
            trials_name: Identifier of the trials entry to remove.

        Returns:
            None
        """
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
        """Join additional columns into the segments table.

        Args:
            right_table: InfoTable/DataFrame or path, or a string key referring to a registered manifest.
            column_names: Columns to add; defaults to all columns.
            on: Column(s) in segments used for the join.
            right_on: Column(s) in the right table used for the join.
            remove_missing: If True, drop segments with missing join keys.
            create_class_info: If True, build ClassInfo tables for newly added columns.

        Returns:
            None
        """
        if isinstance(right_table, (str, Path)):
            file_path = Path(right_table)
            if file_path.is_file():
                right_table = InfoTable.load(file_path)
            else:
                if right_table == "recordings":
                    right_table = self.recordings()
                elif right_table == "images":
                    right_table = self.images()
                elif right_table == "videos":
                    right_table = self.videos()
                elif right_table in self.features_keys():
                    right_table = self.features_value(right_table)
                elif right_table in self.vads_keys():
                    right_table = self.vads_value(right_table)
                elif right_table in self.diarizations_keys():
                    right_table = self.diarizations_value(right_table)
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
        """Create ClassInfo tables from columns in the segments table.

        Args:
            column_names: Column name or list of column names to convert into ClassInfo tables.

        Returns:
            None
        """
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
        """Drop orphaned entries across manifests based on current segments.

        Args:
            rebuild_class_idx: If True, rebuild integer class indices after filtering.

        Returns:
            None
        """

        if self.has_recordings:
            rec_ids = self.segments().recording()
            self._recordings = self.recordings().filter(
                lambda df: df["id"].isin(rec_ids)
            )

        if self.has_images:
            im_ids = self.segments().image()
            self._images = self.images().filter(lambda df: df["id"].isin(im_ids))

        if self.has_videos:
            vid_ids = self.segments().video()
            self._videos = self.videos().filter(lambda df: df["id"].isin(vid_ids))

        ids = self.segments()["id"].values
        for k, table in self.features():
            self._features[k] = table.filter(lambda df: df["id"].isin(ids))

        for k, table in self.vads():
            self._vads[k] = table.filter(lambda df: df["id"].isin(ids))

        for k, table in self.diarizations():
            self._diarizations[k] = table.filter(lambda df: df["id"].isin(ids))

        remove_keys = []
        for k, table in self.classes():
            if k in self.segments():
                class_ids = self.segments()[k].unique()
                self._classes[k] = table.filter(lambda df: df["id"].isin(class_ids))
                if rebuild_class_idx:
                    self._classes[k].add_class_idx()
            else:
                remove_keys.append(k)

        for k in remove_keys:
            self.remove_classes(k)

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
        """Create a trials list and cohort split from a subset of segments.

        Args:
            segments: SegmentSet to sample from.
            num_tar_trials: Number of target trials to generate.
            num_trial_speakers: Number of speakers to include in trials.
            seed: Random seed for reproducibility.

        Returns:
            Tuple[TrialKey, EnrollmentMap, SegmentSet]: Trials, enrollments, and cohort segments.
        """
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
        """Split dataset into a trial subset and a cohort subset for QMF training.

        Args:
            num_1k_tar_trials: Target trials expressed in thousands (e.g., 10 -> 10k trials).
            num_trial_speakers: Number of speakers to use for trials.
            intra_gender: If True, build trials separately within each gender.
            trials_name: Name used to store trials in the returned dataset.
            seed: Random seed for reproducibility.

        Returns:
            Tuple[HyperDataset, HyperDataset]: Dataset with trials/enrollments and dataset with cohort only.
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
        """Remove segments shorter than a given length.

        Args:
            min_length: Minimum allowed duration.
            length_name: Column to compare against ``min_length``.

        Returns:
            None
        """
        segments = self.segments()
        self._segments = segments.filter(lambda df: df[length_name] >= min_length)
        self.clean()

    def remove_classes_few_segments(
        self,
        class_name: str,
        min_segs: int,
        rebuild_idx: bool = False,
    ):
        """Drop classes with fewer than ``min_segs`` segments.

        Args:
            class_name: Column name representing the class label.
            min_segs: Minimum number of segments required to keep a class.
            rebuild_idx: If True, rebuild class indices after filtering.

        Returns:
            None
        """
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
        max_segs: Union[int, None],
        rebuild_idx: bool = False,
    ):
        """Drop classes with too few or too many segments.

        Args:
            class_name: Column name representing the class label.
            min_segs: Minimum number of segments required to keep a class.
            max_segs: Maximum number of segments allowed to keep a class; None to ignore.
            rebuild_idx: If True, rebuild class indices after filtering.

        Returns:
            None
        """
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

    def remove_class_ids(
        self,
        class_name: str,
        class_ids: List[str],
        remove_na: bool,
        rebuild_idx: bool = False,
    ):
        """Remove specific class ids (and optionally NaNs) from the dataset.

        Args:
            class_name: Column name representing the class label.
            class_ids: List of class identifiers to drop.
            remove_na: If True, drop rows with missing class labels.
            rebuild_idx: If True, rebuild class indices after filtering.

        Returns:
            None
        """
        segments = self.segments()
        if remove_na:
            self._segments = segments.dropna(subset=[class_name], inplace=True)

        if class_ids is not None:
            self._segments = segments.filter(lambda df: ~df[class_name].isin(class_ids))
        self.clean()
        if rebuild_idx:
            class_info = self.classes_value(class_name)
            class_info.add_class_idx()

    def filter_by_segments(
        self,
        segments: Union[SegmentSet, List[str]],
        rebuild_class_idx: bool = False,
        keep: bool = True,
    ):
        """Filter dataset by a list of segment ids or a SegmentSet.

        Args:
            segments: Segment ids or SegmentSet to define the filter.
            rebuild_class_idx: If True, rebuild class indices after filtering.
            keep: If True, keep the provided ids; otherwise drop them.

        Returns:
            None
        """

        if isinstance(segments, SegmentSet):
            segment_ids = segments["id"]
        else:
            segment_ids = segments

        segments = self.segments()
        self._segments = segments.filter(items=segment_ids, by="id", keep=keep)
        self.clean(rebuild_class_idx=rebuild_class_idx)

    def filter_by_segments_predicate(
        self,
        predicate: str,
        rebuild_class_idx: bool = False,
        keep: bool = True,
    ):
        """Filter dataset by an expression evaluated on the segments table.

        Args:
            predicate: Query string passed to SegmentSet.filter.
            rebuild_class_idx: If True, rebuild class indices after filtering.
            keep: If True, keep rows matching predicate; otherwise drop them.

        Returns:
            None
        """

        segments = self.segments()
        self._segments = segments.filter(predicate=predicate, keep=keep)
        self.clean(rebuild_class_idx=rebuild_class_idx)

    def filter_by_classes(
        self,
        class_name: str,
        classes: Union[ClassInfo, List[str]],
        remove_na: bool,
        rebuild_idx: bool = False,
        keep: bool = True,
    ):
        """Filter dataset by class membership.

        Args:
            class_name: Column name representing the class label.
            classes: ClassInfo object or list of class ids to keep/drop.
            remove_na: If True, drop rows with missing class labels.
            rebuild_idx: If True, rebuild class indices after filtering.
            keep: If True, retain matching classes; otherwise drop them.

        Returns:
            None
        """
        segments = self.segments()
        if isinstance(classes, ClassInfo):
            class_ids = classes["id"]
        else:
            class_ids = classes

        if remove_na:
            self._segments = segments.dropna(subset=[class_name], inplace=True)

        self._segments = segments.filter(
            lambda df: df[class_name].isin(class_ids), keep=keep
        )
        self.clean()
        if rebuild_idx:
            class_info = self.classes_value(class_name)
            class_info.add_class_idx()

    def filter_by_classes_and_enrollments(
        self,
        class_name: str,
        classes: Union[ClassInfo, List[str]],
        enrollment_name: str,
        enrollments: EnrollmentMap,
        remove_na: bool,
        rebuild_idx: bool = False,
        keep: bool = True,
    ):
        """Filter dataset by class membership and enrollment ids, updating trials too.

        Args:
            class_name: Column name representing the class label.
            classes: ClassInfo object or list of class ids to keep/drop.
            enrollment_name: Enrollment map key to filter.
            enrollments: Enrollment map providing ids to retain/drop.
            remove_na: If True, drop rows with missing class labels.
            rebuild_idx: If True, rebuild class indices after filtering.
            keep: If True, retain matching classes/enrollments; otherwise drop them.

        Returns:
            None
        """
        segments = self.segments()
        if isinstance(classes, ClassInfo):
            class_ids = classes["id"]
        else:
            class_ids = classes

        if remove_na:
            self._segments = segments.dropna(subset=[class_name], inplace=True)

        self._segments = segments.filter(
            lambda df: df[class_name].isin(class_ids), keep=keep
        )
        model_ids = np.unique(np.unique(enrollments["id"]))
        if self._enrollments is not None and enrollment_name in self._enrollments:
            my_enrollments = self.enrollments_value(enrollment_name)
            self.enrollments[enrollment_name] = my_enrollments.filter(
                items=model_ids, keep=keep
            )

        for k, key in self.trials():
            key = key.filter_by_model(model_ids, keep=keep, raise_missing=False)
            self._trials[k] = key

        self.clean()
        if rebuild_idx:
            class_info = self.classes_value(class_name)
            class_info.add_class_idx()

    def rebuild_class_idx(self, class_name: str):
        """Recompute integer class indices for a given class info table.

        Args:
            class_name: Name of the class info table.

        Returns:
            None
        """
        class_info = self.classes_value(class_name)
        class_info.add_class_idx()

    def _segments_split(self, val_prob: float, rng: np.random.Generator):
        """Randomly split segments into train/validation folds.

        Args:
            val_prob: Fraction of segments to place in validation.
            rng: Random generator to use for permutation.

        Returns:
            Tuple[SegmentSet, SegmentSet]: Training and validation segments.
        """
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
        """Split ensuring each joint class combination appears in both splits.

        Args:
            val_prob: Fraction of samples per class to place in validation.
            joint_classes: Columns defining joint class membership.
            min_train_samples: Minimum training samples per joint class.
            rng: Random generator to use for permutation.

        Returns:
            Tuple[SegmentSet, SegmentSet]: Training and validation segments.
        """
        segments = self.segments()
        classes = segments[joint_classes].apply("-".join, axis=1)
        u_classes, class_ids = np.unique(classes, return_inverse=True)
        train_mask = np.zeros(len(segments), dtype=bool)
        # kk = 0
        for c_id in range(len(u_classes)):
            idx = (class_ids == c_id).nonzero()[0]
            count = len(idx)
            p = rng.permutation(count)
            num_train = max(
                int(round((1 - val_prob) * count)), min(min_train_samples, count)
            )
            # kk += count - num_train
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
        """Split ensuring disjoint sets of classes between train and validation.

        Args:
            val_prob: Fraction of segments to place in validation.
            disjoint_classes: Columns defining mutually exclusive classes.
            rng: Random generator to use for permutation.

        Returns:
            Tuple[SegmentSet, SegmentSet]: Training and validation segments.
        """
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
        """Placeholder for joint/disjoint class split logic."""
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
        """Create train/validation dataset splits with optional class constraints.

        Args:
            val_prob: Fraction of segments to place in validation.
            joint_classes: Columns that must appear in both splits.
            disjoint_classes: Columns that must not overlap between splits.
            min_train_samples: Minimum samples per joint class when ``joint_classes`` is used.
            seed: Random seed for reproducibility.

        Returns:
            Tuple[HyperDataset, HyperDataset]: Train and validation datasets.
        """
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

    def _segments_split_folds(self, num_folds: int, rng: np.random.Generator):
        """Randomly split segments into ``num_folds`` folds.

        Args:
            num_folds: Number of folds to create.
            rng: Random generator to use for permutation.

        Returns:
            Tuple[List[SegmentSet], List[SegmentSet]]: Training and test folds.
        """
        segments = self.segments()
        p = rng.permutation(len(segments))
        segs_per_fold = len(p) // num_folds
        start = 0
        train_folds = []
        test_folds = []
        for i in range(num_folds):
            if i < num_folds - 1:
                fold_idx = p[start : start + segs_per_fold]
                start += segs_per_fold
            else:
                fold_idx = p[start:]

            test_fold_segs = segments.filter(iiindex=fold_idx, keep=True)
            test_fold_segs.sort()
            train_fold_segs = segments.filter(iiindex=fold_idx, keep=False)
            train_fold_segs.sort()
            test_folds.append(test_fold_segs)
            train_folds.append(train_fold_segs)

        return train_folds, test_folds

    def _segments_split_folds_joint_classes(
        self,
        num_folds: int,
        joint_classes: List[str],
        rng: np.random.Generator,
    ):
        """Create folds while keeping each joint class combination in every fold.

        Args:
            num_folds: Number of folds to create.
            joint_classes: Columns defining joint class membership.
            rng: Random generator to use for permutation.

        Returns:
            Tuple[List[SegmentSet], List[SegmentSet]]: Training and test folds.
        """
        segments = self.segments()
        classes = segments[joint_classes].apply("-".join, axis=1)
        u_classes, class_ids, class_counts = np.unique(
            classes, return_inverse=True, return_counts=True
        )
        train_folds = []
        test_folds = []
        starts = np.zeros((len(u_classes)), dtype=int)
        class_segs_per_fold = class_counts // num_folds
        permutations = []
        for c_id in range(len(u_classes)):
            permutations.append(rng.permutation(class_counts[c_id]))

        for i in range(num_folds):
            test_idx = []
            for c_id in range(len(u_classes)):
                idx = (class_ids == c_id).nonzero()[0]
                idx = idx[permutations[c_id]]
                if i < num_folds - 1:
                    idx_fold = idx[
                        starts[c_id] : starts[c_id] + class_segs_per_fold[c_id]
                    ]
                    starts[c_id] = starts[c_id] + class_segs_per_fold[c_id]
                else:
                    idx_fold = idx[starts[c_id] :]

                assert len(idx_fold) < len(
                    idx
                ), f"{u_classes[i]} with {len(idx)} samples doesn't have training samples in fold {i}"
                test_idx.append(idx_fold)

            test_idx = np.concatenate(test_idx, axis=0)
            test_segs = segments.filter(iindex=test_idx)
            test_segs.sort()

            train_segs = segments.filter(iindex=test_idx, keep=False)
            train_segs.sort()
            test_folds.append(test_segs)
            train_folds.append(train_segs)

        return train_folds, test_folds

    def _segments_split_folds_disjoint_classes(
        self,
        num_folds: float,
        disjoint_classes: List[str],
        rng: np.random.Generator,
    ):
        """Create folds such that class groups are disjoint across folds.

        Args:
            num_folds: Number of folds to create.
            disjoint_classes: Columns defining mutually exclusive classes.
            rng: Random generator to use for permutation.

        Returns:
            Tuple[List[SegmentSet], List[SegmentSet]]: Training and test folds.
        """
        segments = self.segments()
        classes = segments[disjoint_classes].apply("-".join, axis=1)
        u_classes, class_ids = np.unique(classes, return_inverse=True)
        p = rng.permutation(len(u_classes))
        class_ids = p[class_ids]
        classes_per_fold = len(u_classes) // num_folds
        train_folds = []
        test_folds = []
        start = 0
        for i in range(num_folds):
            if i < num_folds - 1:
                test_mask = np.logical_and(
                    class_ids >= start, class_ids < start + classes_per_fold
                )
                start += classes_per_fold
            else:
                test_mask = class_ids >= start

            test_idx = test_mask.nonzero()[0]
            test_segs = segments.filter(iindex=test_idx, keep=True)
            test_segs.sort()
            train_segs = segments.filter(iindex=test_idx, keep=False)
            train_segs.sort()
            test_folds.append(test_segs)
            train_folds.append(train_segs)

        return train_folds, test_folds

    def _segments_split_folds_joint_and_disjoint_classes(
        self,
        num_folds: int,
        joint_classes: List[str],
        disjoint_classes: List[str],
        rng: np.random.Generator,
    ):
        """Create folds balancing both joint and disjoint class constraints."""
        segments = self.segments()
        jclasses = segments[joint_classes].apply("-".join, axis=1)
        u_jclasses, jclass_ids = np.unique(jclasses, return_inverse=True)
        dclasses = segments[disjoint_classes].apply("-".join, axis=1)
        u_dclasses, dclass_ids = np.unique(
            dclasses,
            return_inverse=True,
        )
        counts = np.zeros((len(u_dclasses), len(u_jclasses)), dtype=int)
        for i in range(len(u_dclasses)):
            jclass_ids_i = jclass_ids[dclass_ids == i]
            for j in range(len(u_jclasses)):
                counts[i, j] = np.sum(jclass_ids_i == j)

        available = {i for i in range(len(u_dclasses))}
        sel_class = rng.choice(len(u_dclasses))
        available.remove(sel_class)
        p = np.zeros(len(u_dclasses), dtype=int)
        counts_acc = counts[sel_class]
        for i in range(1, len(u_dclasses)):
            best_j = -1
            best_entropy = 0
            for j in available:
                counts_ij = counts_acc + counts[j]
                p_ij = counts_ij / counts_ij.sum()
                entropy_ij = -np.sum(p_ij * np.log(p_ij + 1e-5))
                if entropy_ij > best_entropy:
                    best_entropy = entropy_ij
                    best_j = j
            p[best_j] = i
            available.remove(best_j)
            counts_acc += counts[best_j]

        dclass_ids = p[dclass_ids]
        classes_per_fold = len(u_dclasses) // num_folds
        train_folds = []
        test_folds = []
        start = 0
        for i in range(num_folds):
            if i < num_folds - 1:
                test_mask = np.logical_and(
                    dclass_ids >= start, dclass_ids < start + classes_per_fold
                )
                start += classes_per_fold
            else:
                test_mask = dclass_ids >= start

            test_idx = test_mask.nonzero()[0]
            test_segs = segments.filter(iindex=test_idx, keep=True)
            test_segs.sort()
            train_segs = segments.filter(iindex=test_idx, keep=False)
            train_segs.sort()
            test_folds.append(test_segs)
            train_folds.append(train_segs)

        return train_folds, test_folds

    def split_folds(
        self,
        num_folds: int,
        joint_classes: Optional[List[str]] = None,
        disjoint_classes: Optional[List[str]] = None,
        seed: int = 11235813,
    ):
        """Create cross-validation folds with optional class constraints.

        Args:
            num_folds: Number of folds to create.
            joint_classes: Columns that must appear across all folds.
            disjoint_classes: Columns that must be disjoint across folds.
            seed: Random seed for reproducibility.

        Returns:
            Tuple[List[HyperDataset], List[HyperDataset]]: Training and test datasets per fold.
        """
        rng = np.random.default_rng(seed)
        if joint_classes is None and disjoint_classes is None:
            train_segs, test_segs = self._segments_folds_split(num_folds, rng)
        elif joint_classes is not None and disjoint_classes is None:
            train_segs, test_segs = self._segments_split_folds_joint_classes(
                num_folds,
                joint_classes,
                rng,
            )
        elif joint_classes is None and disjoint_classes is not None:
            train_segs, test_segs = self._segments_split_folds_disjoint_classes(
                num_folds,
                disjoint_classes,
                rng,
            )
        else:
            train_segs, test_segs = (
                self._segments_split_folds_joint_and_disjoint_classes(
                    num_folds,
                    joint_classes,
                    disjoint_classes,
                    rng,
                )
            )

        train_folds = []
        test_folds = []
        for train_segs_i, test_segs_i in zip(train_segs, test_segs):
            train_fold = self.clone()
            train_fold.set_segments(train_segs_i)
            train_fold.clean()

            test_fold = self.clone()
            test_fold.set_segments(test_segs_i)
            test_fold.clean()

            train_folds.append(train_fold)
            test_folds.append(test_fold)

        return train_folds, test_folds

    @classmethod
    def merge(cls, datasets):
        """Concatenate multiple HyperDataset objects into one.

        Args:
            datasets: Iterable of HyperDataset instances to merge.

        Returns:
            HyperDataset: New dataset containing concatenated manifests where possible.
        """
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
            if dset.has_recordings:
                recs_i = dset.recordings(keep_loaded=False)
                recordings.append(recs_i)

        if recordings:
            recordings = RecordingSet.cat(recordings)
            dataset.set_recordings(recordings)

        images = []
        for dset in datasets:
            if dset.has_images:
                ims_i = dset.images(keep_loaded=False)
                images.append(ims_i)

        if images:
            images = ImageSet.cat(images)
            dataset.set_images(images)

        videos = []
        for dset in datasets:
            if dset.has_videos:
                vids_i = dset.videos(keep_loaded=False)
                videos.append(vids_i)

        if videos:
            videos = VideoSet.cat(videos)
            dataset.set_videos(videos)

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

        vads_keys = []
        for dset in datasets:
            vads_dset = list(dset.vads_keys())
            vads_keys.extend(vads_dset)

        vads_keys = list(set(vads_keys))
        for key in vads_keys:
            vads = []
            for dset in datasets:
                if key in dset.vads_keys():
                    vads_key = dset.vads_value(key, keep_loaded=False)
                    vads.append(vads_key)

            vads = VADSet.cat(vads)
            dataset.add_vads(vads_name=key, vads=vads)

        diarizations_keys = []
        for dset in datasets:
            diarizations_dset = list(dset.diarizations_keys())
            diarizations_keys.extend(diarizations_dset)

        diarizations_keys = list(set(diarizations_keys))
        for key in diarizations_keys:
            diarizations = []
            for dset in datasets:
                if key in dset.diarizations_keys():
                    diarizations_key = dset.diarizations_value(key, keep_loaded=False)
                    diarizations.append(diarizations_key)

            diarizations = DiarizationSet.cat(diarizations)
            dataset.add_diarizations(diarizations_name=key, diarizations=diarizations)

        # TODO: merge enrollments and trials
        # Usually you don't need that
        return dataset

    def add_classes_from_segments(self, class_names: Union[str, List[str]]):
        """Build ClassInfo objects from columns already present in segments.

        Args:
            class_names: Column name or list of column names to convert to ClassInfo.

        Returns:
            None
        """
        if isinstance(class_names, str):
            class_names = [class_names]

        for col in class_names:
            if col not in self._classes:
                logging.info(f"Building ClassInfo for column '{col}' from segments")
                segment_values = self.segments(keep_loaded=True)[col]
                segment_values = segment_values[pd.notna(segment_values)]
                df = pd.DataFrame({"id": np.unique(segment_values)})
                class_info = ClassInfo(df)
                self.add_classes(col, class_info)

    @classmethod
    def from_recordings(cls, recordings: Union[RecordingSet, PathLike]):
        """Create a dataset from recordings when no segmentation exists.

        Args:
            recordings: RecordingSet object or path to a RecordingSet manifest.

        Returns:
            HyperDataset: Dataset whose segments mirror the recordings table.
        """
        if isinstance(recordings, (str, Path)):
            recordings = RecordingSet.load(recordings)

        seg_df = recordings[["id", "duration"]]
        segments = SegmentSet(seg_df)
        dataset = cls(segments=segments, recordings=recordings)
        return dataset

    @classmethod
    def from_segments(
        cls,
        segments: Union[SegmentSet, PathLike],
        recordings: Optional[Union[RecordingSet, PathLike]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Create a dataset from a SegmentSet with optional recordings and classes.

        Args:
            segments: SegmentSet object or path to a segments manifest.
            recordings: Optional RecordingSet object or path.
            class_names: Optional class columns to convert into ClassInfo tables.

        Returns:
            HyperDataset: Dataset built from the provided manifests.
        """
        if isinstance(segments, (str, Path)):
            segments = SegmentSet.load(segments)

        if recordings is not None:
            if isinstance(recordings, (str, Path)):
                recordings = RecordingSet.load(recordings)
            else:
                if "duration" not in segments:
                    segments["duration"] = recordings.loc[segments["id"], "duration"]

        classes = None
        if class_names is not None:
            classes = {}
            for class_name in class_names:
                if class_name in segments:
                    class_values = segments[class_name]
                    class_values = class_values[pd.notna(class_values)]
                    uniq_classes = np.unique(class_values)
                    classes[class_name] = ClassInfo(pd.DataFrame({"id": uniq_classes}))

        dataset = cls(segments=segments, recordings=recordings, classes=classes)
        return dataset

    @classmethod
    def from_lhotse(
        cls,
        cuts: Optional[Union[lhotse.CutSet, PathLike]] = None,
        recordings: Optional[Union[lhotse.RecordingSet, PathLike]] = None,
        supervisions: Optional[Union[lhotse.SupervisionSet, PathLike]] = None,
    ):
        """Create a dataset from Lhotse cuts or from recordings + supervisions.

        Args:
            cuts: Lhotse CutSet object or path to a CutSet manifest.
            recordings: Optional Lhotse RecordingSet object or path.
            supervisions: Optional Lhotse SupervisionSet object or path.

        Returns:
            HyperDataset: Dataset derived from the Lhotse manifests.
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
        """Create a dataset from a Kaldi-style data directory.

        Args:
            kaldi_data_dir: Path to a Kaldi data directory.

        Returns:
            HyperDataset: Dataset populated from Kaldi manifests.
        """
        kaldi_data_dir = Path(kaldi_data_dir)

        kaldi_files = ["utt2lang", "utt2dur", "text"]
        attributes = ["language", "duration", "text"]

        k_file = kaldi_data_dir / "utt2spk"
        from .utt2info import Utt2Info

        utt2spk = Utt2Info.load(k_file)
        df_segs = pd.DataFrame({"id": utt2spk.key, "speaker": utt2spk.info})
        segments = SegmentSet(df_segs)
        del utt2spk

        for k_file, att in zip(kaldi_files, attributes):
            k_file = kaldi_data_dir / k_file
            if k_file.is_file():
                if att == "text":
                    u2i = SCPList.load(k_file)
                    info = u2i.file_path
                else:
                    u2i = Utt2Info.load(k_file)
                    if len(u2i.info.shape) > 1:
                        u2i.info = u2i.info[:, 0]
                    info = u2i.info

                segments.loc[u2i.key, att] = info

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

    def append_seg_suffix(self, seg_suffix: str):
        """Append a suffix to all segment ids (and aligned manifest ids)."""
        segments = self.segments(keep_loaded=True)
        segments["id"] = segments["id"].apply(lambda x: x + seg_suffix)
        if self.has_recordings and "recording" not in segments:
            recs = self.recordings(keep_loaded=True)
            recs["id"] = recs["id"].apply(lambda x: x + seg_suffix)

        if self.has_images and "image" not in segments:
            ims = self.images(keep_loaded=True)
            ims["id"] = ims["id"].apply(lambda x: x + seg_suffix)

        if self.has_videos and "video" not in segments:
            vids = self.videos(keep_loaded=True)
            vids["id"] = vids["id"].apply(lambda x: x + seg_suffix)

        for key, feats in self.features(keep_loaded=True):
            feats["id"] = feats["id"].apply(lambda x: x + seg_suffix)

        for key, vad in self.vads(keep_loaded=True):
            vad["id"] = vad["id"].apply(lambda x: x + seg_suffix)

    def cat_segments(
        self,
        group_by: Union[str, List[str]],
        max_duration: Optional[float] = None,
        inplace: bool = False,
    ):
        """Concatenate segments within groups and rebuild recordings with a sox pipe.

        Args:
            group_by: Column name or list of columns to define concatenation groups.
            max_duration: Maximum duration in seconds for each concatenated segment.
                When exceeded, a new concatenated segment is started.
            inplace: If True, modify the dataset in place; otherwise return a clone.

        Returns:
            HyperDataset: Dataset with concatenated segments and recordings.
        """
        # Normalize group_by to a list for uniform processing.
        if isinstance(group_by, str):
            group_by = [group_by]

        # Require at least one grouping column.
        if not group_by:
            raise ValueError("cat_segments requires at least one grouping column")

        # Validate max_duration when provided.
        if max_duration is not None and max_duration <= 0:
            raise ValueError("max_duration must be positive when provided")

        # Load segments into memory to perform concatenation.
        segments = self.segments(keep_loaded=True)
        # Ensure requested grouping columns exist.
        missing_cols = [col for col in group_by if col not in segments]
        if missing_cols:
            raise ValueError(
                f"group_by columns not found in segments: {', '.join(missing_cols)}"
            )

        # Disallow concatenation when segments have non-zero or missing starts.
        if "start" in segments:
            start_vals = segments["start"]
            if start_vals.isna().any() or (np.abs(start_vals) > 1e-9).any():
                raise ValueError(
                    "cat_segments requires start==0 for all segments (no offsets allowed)"
                )

        # Require recordings to build concatenated storage paths.
        if not self.has_recordings:
            raise ValueError("cat_segments requires a recordings table")

        # Load recordings and ensure none are already pipes.
        recordings = self.recordings(keep_loaded=True)
        storage_paths = recordings["storage_path"].astype(str).str.strip()
        if storage_paths.str.endswith("|").any():
            raise ValueError(
                "cat_segments does not allow recordings with pipe storage_path entries"
            )
        if "sample_freq" not in recordings:
            raise ValueError("cat_segments requires recordings to have sample_freq")

        # Resolve per-segment durations (from segments or recordings).
        if "duration" in segments:
            seg_durations = segments["duration"].astype(float)
        else:
            if "duration" not in recordings:
                raise ValueError(
                    "cat_segments requires segment durations or recordings durations"
                )
            rec_ids = segments.recording()
            seg_durations = recordings.loc[rec_ids, "duration"].astype(float)
        # Ensure we have valid durations everywhere.
        if seg_durations.isna().any():
            raise ValueError("cat_segments requires non-missing durations")

        # Work on a copy of the segments DataFrame to avoid side-effects.
        seg_df = segments.df.copy()
        # Cache duration and recording id for concatenation logic.
        seg_df["_cat_duration"] = seg_durations.values
        seg_df["_cat_rec_id"] = segments.recording().values

        # Aggregate per-column values for concatenated segments.
        def _agg_value(series: pd.Series, dtype):
            if pd.api.types.is_bool_dtype(series):
                return series.iloc[0] if series.nunique(dropna=False) == 1 else pd.NA
            if pd.api.types.is_numeric_dtype(series):
                mean_val = series.astype(float).mean()
                if pd.isna(mean_val):
                    return pd.NA
                if pd.api.types.is_integer_dtype(dtype):
                    return int(round(mean_val))
                return mean_val
            return series.iloc[0] if series.nunique(dropna=False) == 1 else pd.NA

        # Ensure fields like sample_freq are constant within a concatenation group.
        def _unique_or_error(series: pd.Series, name: str):
            values = series.dropna().unique()
            if len(values) == 0:
                return np.nan
            if len(values) == 1:
                return values[0]
            raise ValueError(
                f"cat_segments requires a single {name} within each concatenation group"
            )

        # Columns that are handled explicitly during concatenation.
        skip_seg_cols = {
            "id",
            "recording",
            "start",
            "duration",
            "_cat_duration",
            "_cat_rec_id",
        }
        # Columns to aggregate over when building new segment rows.
        seg_cols = [col for col in seg_df.columns if col not in skip_seg_cols]
        seg_col_dtypes = {col: seg_df[col].dtype for col in seg_cols}

        # Track whether these columns exist to preserve schema.
        has_recording_col = "recording" in segments
        has_start_col = "start" in segments

        # Accumulate new segments and recordings rows.
        new_segments_rows = []
        new_recordings_rows = []

        # Validate that all segment recording ids exist in the recordings table.
        rec_df = recordings.df
        rec_col_dtypes = {col: rec_df[col].dtype for col in rec_df.columns}
        missing_rec_ids = np.setdiff1d(seg_df["_cat_rec_id"].unique(), rec_df.index)
        if len(missing_rec_ids) > 0:
            raise ValueError(
                "cat_segments found segments with missing recordings: "
                + ", ".join(map(str, missing_rec_ids[:10]))
            )

        # Build concatenated segments for each group.
        for group_key, group in seg_df.groupby(group_by, sort=False, dropna=False):
            # Preserve the original order as encountered.
            group = group.sort_index()
            # Build the base name from the group-by values.
            if len(group_by) == 1:
                if isinstance(group_key, tuple) and len(group_key) == 1:
                    base_name = str(group_key[0])
                else:
                    base_name = str(group_key)
            else:
                base_name = "-".join(str(value) for value in group_key)
            # Track which rows are in the current concatenation chunk.
            chunk_indices = []
            # Track the duration in the current chunk.
            chunk_duration = 0.0
            # Number chunks per group starting at 1.
            chunk_idx = 0

            # Iterate over segments in the group, building chunks.
            for row_idx, row in group.iterrows():
                row_dur = row["_cat_duration"]
                # If adding this segment exceeds max_duration, finalize the chunk.
                if (
                    max_duration is not None
                    and chunk_indices
                    and chunk_duration + row_dur > max_duration
                ):
                    # Increment chunk index and slice chunk data.
                    chunk_idx += 1
                    chunk = seg_df.loc[chunk_indices]
                    # Create a new segment id with the group base name and chunk index.
                    new_id = f"{base_name}-{chunk_idx:04d}"

                    # Build the new segment row with summed duration.
                    seg_row = {"id": new_id, "duration": chunk["_cat_duration"].sum()}
                    if has_start_col:
                        seg_row["start"] = 0.0
                    if has_recording_col:
                        seg_row["recording"] = new_id

                    # Aggregate remaining segment columns.
                    for col in seg_cols:
                        if col == "transcript":
                            transcripts = chunk[col].dropna().astype(str).tolist()
                            if transcripts:
                                merged = transcripts[0]
                                for part in transcripts[1:]:
                                    if merged.rstrip().endswith("."):
                                        merged = f"{merged} {part}"
                                    else:
                                        merged = f"{merged}. {part}"
                                seg_row[col] = merged
                            else:
                                seg_row[col] = pd.NA
                        else:
                            seg_row[col] = _agg_value(chunk[col], seg_col_dtypes[col])

                    # Store the new segment row.
                    new_segments_rows.append(seg_row)

                    # Build the new recording row, using a pipe only when concatenating.
                    rec_chunk = rec_df.loc[chunk["_cat_rec_id"].values]
                    use_pipe = len(rec_chunk) > 1
                    rec_row = {
                        "id": new_id,
                        "storage_path": (
                            "sox "
                            + " ".join(rec_chunk["storage_path"].astype(str))
                            + " -t wav - |"
                            if use_pipe
                            else str(rec_chunk["storage_path"].iloc[0])
                        ),
                    }
                    # Sum or fallback duration for recordings.
                    if "duration" in rec_chunk:
                        rec_row["duration"] = rec_chunk["duration"].sum()
                    else:
                        rec_row["duration"] = seg_row["duration"]

                    # Preserve sample_freq when consistent.
                    if "sample_freq" in rec_chunk:
                        rec_row["sample_freq"] = _unique_or_error(
                            rec_chunk["sample_freq"], "sample_freq"
                        )

                    # Aggregate other recording columns.
                    skip_rec_cols = {"id", "storage_path", "duration", "sample_freq"}
                    for col in rec_chunk.columns:
                        if col in skip_rec_cols:
                            continue
                        rec_row[col] = _agg_value(rec_chunk[col], rec_col_dtypes[col])

                    # Store the new recording row.
                    new_recordings_rows.append(rec_row)

                    # Reset chunk accumulators.
                    chunk_indices = []
                    chunk_duration = 0.0

                # Add current segment to the chunk.
                chunk_indices.append(row_idx)
                chunk_duration += row_dur

            # Finalize the last chunk in the group.
            if chunk_indices:
                chunk_idx += 1
                chunk = seg_df.loc[chunk_indices]
                new_id = f"{base_name}-{chunk_idx:04d}"

                # Build the new segment row with summed duration.
                seg_row = {"id": new_id, "duration": chunk["_cat_duration"].sum()}
                if has_start_col:
                    seg_row["start"] = 0.0
                if has_recording_col:
                    seg_row["recording"] = new_id

                # Aggregate remaining segment columns.
                for col in seg_cols:
                    if col == "transcript":
                        transcripts = chunk[col].dropna().astype(str).tolist()
                        if transcripts:
                            merged = transcripts[0]
                            for part in transcripts[1:]:
                                if merged.rstrip().endswith("."):
                                    merged = f"{merged} {part}"
                                else:
                                    merged = f"{merged}. {part}"
                            seg_row[col] = merged
                        else:
                            seg_row[col] = pd.NA
                    else:
                        seg_row[col] = _agg_value(chunk[col], seg_col_dtypes[col])

                # Store the new segment row.
                new_segments_rows.append(seg_row)

                # Build the new recording row, using a pipe only when concatenating.
                rec_chunk = rec_df.loc[chunk["_cat_rec_id"].values]
                use_pipe = len(rec_chunk) > 1
                rec_row = {
                    "id": new_id,
                    "storage_path": (
                        "sox "
                        + " ".join(rec_chunk["storage_path"].astype(str))
                        + " -t wav - |"
                        if use_pipe
                        else str(rec_chunk["storage_path"].iloc[0])
                    ),
                }
                # Sum or fallback duration for recordings.
                if "duration" in rec_chunk:
                    rec_row["duration"] = rec_chunk["duration"].sum()
                else:
                    rec_row["duration"] = seg_row["duration"]

                # Preserve sample_freq when consistent.
                if "sample_freq" in rec_chunk:
                    rec_row["sample_freq"] = _unique_or_error(
                        rec_chunk["sample_freq"], "sample_freq"
                    )

                # Aggregate other recording columns.
                skip_rec_cols = {"id", "storage_path", "duration", "sample_freq"}
                for col in rec_chunk.columns:
                    if col in skip_rec_cols:
                        continue
                    rec_row[col] = _agg_value(rec_chunk[col], rec_col_dtypes[col])

                # Store the new recording row.
                new_recordings_rows.append(rec_row)

        # Build dataframes from the accumulated rows.
        new_segments_df = pd.DataFrame(new_segments_rows)
        new_recordings_df = pd.DataFrame(new_recordings_rows)

        # Track required segment columns for column-pruning.
        required_seg_cols = {"id", "duration"}
        if has_start_col:
            required_seg_cols.add("start")
        if has_recording_col:
            required_seg_cols.add("recording")

        # Drop segment columns that are entirely NA after aggregation.
        drop_seg_cols = [
            col
            for col in new_segments_df.columns
            if col not in required_seg_cols and new_segments_df[col].isna().all()
        ]
        if drop_seg_cols:
            new_segments_df.drop(columns=drop_seg_cols, inplace=True)

        # Drop recording columns that are entirely NA after aggregation.
        drop_rec_cols = [
            col
            for col in new_recordings_df.columns
            if col not in {"id", "storage_path"} and new_recordings_df[col].isna().all()
        ]
        if drop_rec_cols:
            new_recordings_df.drop(columns=drop_rec_cols, inplace=True)

        # Preserve original column order where possible.
        seg_order = [col for col in segments.df.columns if col in new_segments_df.columns]
        for col in new_segments_df.columns:
            if col not in seg_order:
                seg_order.append(col)
        new_segments_df = new_segments_df[seg_order]

        # Wrap in SegmentSet/RecordingSet and fix segment dtypes.
        new_segments = SegmentSet(new_segments_df)
        self._fix_segments_dtypes(new_segments)
        new_recordings = RecordingSet(new_recordings_df)

        # Decide whether to mutate the dataset in place.
        dataset = self if inplace else self.clone()
        # Swap in new manifests and clear file-backed paths.
        dataset._segments = new_segments
        dataset._segments_path = None
        dataset._recordings = new_recordings
        dataset._recordings_path = None

        # Remove any feature sets since ids no longer align.
        for key in list(dataset.features_keys()):
            dataset.remove_features(key)

        # Remove any VADs since ids no longer align.
        for key in list(dataset.vads_keys()):
            dataset.remove_vads(key)

        # Remove any diarizations since ids no longer align.
        for key in list(dataset.diarizations_keys()):
            dataset.remove_diarizations(key)

        # Remove enrollments because segment ids changed.
        enrollment_keys = []
        if dataset._enrollments is not None:
            enrollment_keys = list(dataset._enrollments.keys())
        elif dataset._enrollments_paths is not None:
            enrollment_keys = list(dataset._enrollments_paths.keys())
        for key in enrollment_keys:
            dataset.remove_enrollments(key)

        # Remove trials because segment ids changed.
        trial_keys = []
        if dataset._trials is not None:
            trial_keys = list(dataset._trials.keys())
        elif dataset._trials_paths is not None:
            trial_keys = list(dataset._trials_paths.keys())
        for key in trial_keys:
            dataset.remove_trials(key)

        # Remove class tables if their corresponding segment columns disappeared.
        removed_class_cols = set(drop_seg_cols)
        class_keys = []
        if dataset._classes is not None:
            class_keys = list(dataset._classes.keys())
        elif dataset._classes_paths is not None:
            class_keys = list(dataset._classes_paths.keys())
        for key in class_keys:
            if key in removed_class_cols:
                dataset.remove_classes(key)

        # Re-run consistency cleanup for remaining manifests.
        dataset.clean()
        return dataset

    def sample_random_subsegments(
        self,
        subsegments_per_segment: int = 1,
        min_duration: float = 0.0,
        max_duration: Optional[float] = None,
        seg_suffix: Optional[str] = None,
        random_start: bool = True,
        seed: int = 11235813,
        rng: Optional[np.random.Generator] = None,
        inplace: bool = True,
    ):
        """Sample random subsegments for each segment and optionally apply to the dataset.

        Args:
            subsegments_per_segment: Number of subsegments to draw from each original segment.
            min_duration: Minimum duration of sampled subsegments.
            max_duration: Maximum duration of sampled subsegments; None for full length.
            seg_suffix: Optional suffix to append to new segment ids.
            random_start: If True, choose random start within the segment.
            seed: Seed for the internal random generator if ``rng`` is not provided.
            rng: Optional numpy Generator to control randomness.
            inplace: If True, modify current dataset; otherwise return a cloned dataset.

        Returns:
            HyperDataset: Dataset containing the sampled subsegments.
        """
        segments = self.segments(keep_loaded=True)
        segments = segments.sample_random_subsegments(
            subsegments_per_segment,
            min_duration,
            max_duration,
            seg_suffix,
            random_start,
            seed=seed,
            rng=rng,
        )

        if inplace:
            new_dataset = self
        else:
            new_dataset = self.clone()

        new_dataset._segments = segments
        for k in new_dataset.vads_keys():
            new_dataset.remove_vads(k)

        for k in new_dataset.features_keys():
            new_dataset.remove_features(k)

        for k in new_dataset.diarizations_keys():
            new_dataset.remove_diarizations(k)

        if seg_suffix is not None or subsegments_per_segment > 1:
            new_dataset.remove_enrollments()
            new_dataset.remove_trials()

        return new_dataset


class HypDataset(HyperDataset):
    pass
