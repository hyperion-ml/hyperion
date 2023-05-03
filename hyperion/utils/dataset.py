"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

from .class_info import ClassInfo
from .feature_set import FeatureSet
from .misc import PathLike
from .recording_set import RecordingSet
from .segment_set import SegmentSet


class Dataset:
    """ Class that contains all objects 
        (segments, recordings, features, class_infos) that 
        conform a dataset
    """

    def __init__(
        self,
        segments: SegmentSet,
        classes: Optional[Dict[str, ClassInfo]] = None,
        recordings: Optional[Dict[str, RecordingSet]] = None,
        features: Optional[Dict[str, FeatureSet]] = None,
    ):
        self._segments = segments
        self._classes = classes
        self._recordings = recordings
        self._features = features

    @property
    def segments(self):
        return self._segments

    @property
    def recordings(self):
        return self._recordings

    @property
    def features(self):
        return self._features

    @property
    def classes(self):
        return self._classes

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
        if file_path.is_file():
            return file_path

        return dataset_dir / file_path

    def save(self, dataset_path: PathLike):
        """Saves all the dataset objects.

        Args:
         dataset_path: str/Path indicating directory 
          to save the dataset or .yaml file to save 
          the dataset info.

        """
        dataset_dir, dataset_file = Dataset.resolve_dataset_path(dataset_path)
        dataset = {}
        if self.segments is not None:
            file_name = "segments.csv"
            dataset["segments"] = file_name
            file_path = dataset_dir / file_name
            self.segments.save(file_path)

        if self.recordings is not None:
            file_names = {}
            for k, v in self.recordings.items():
                file_name = k + ".csv"
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                v.save(file_path)

            dataset["recordings"] = file_names

        if self.features is not None:
            file_names = {}
            for k, v in self.features.items():
                file_name = k + ".csv"
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                v.save(file_path)

            dataset["features"] = file_names

        if self.classes is not None:
            file_names = {}
            for k, v in self.classes.items():
                file_name = k + ".csv"
                file_names[k] = file_name
                file_path = dataset_dir / file_name
                v.save(file_path)

            dataset["classes"] = file_names

        with open(dataset_file, "w") as f:
            yaml.dump(dataset, f)

    @classmethod
    def load(cls, dataset_path: PathLike):
        """Loads all the dataset objects.

        Args:
         dataset_path: str/Path indicating directory 
          to save the dataset or .yaml file to save 
          the dataset info.

        """
        dataset_dir, dataset_file = Dataset.resolve_dataset_path(dataset_path)
        with open(dataset_file, "w") as f:
            dataset = yaml.safe_load(f)

        assert "segments" in dataset
        segments = SegmentSet.load(
            Dataset.resolve_file_path(dataset_dir, dataset["segments"])
        )
        classes = None
        recordings = None
        features = None
        if "classes" in dataset:
            classes = {}
            for k, v in dataset["classes"]:
                classes[k] = ClassInfo.load(Dataset.resolve_file_path(dataset_dir, v))

        if "recordings" in dataset:
            recordings = {}
            for k, v in dataset["recordings"]:
                recordings[k] = RecordingSet.load(
                    Dataset.resolve_file_path(dataset_dir, v)
                )

        if "features" in dataset:
            features = {}
            for k, v in dataset["features"]:
                features[k] = FeatureSet.load(Dataset.resolve_file_path(dataset_dir, v))

        return cls(segments, classes, recordings, features)
