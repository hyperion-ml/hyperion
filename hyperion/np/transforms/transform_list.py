"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import h5py
import numpy as np
from typing import Any, Dict, List, Union

from ..hyper_np_model import HyperNPModel
from .cent_whiten import CentWhiten
from .cent_whiten_up import CentWhitenUP
from .gaussianizer import Gaussianizer
from .lda import LDA
from .lnorm import LNorm
from .lnorm_up import LNormUP
from .mvn import MVN
from .nap import NAP
from .nda import NDA
from .pca import PCA


class TransformList(HyperNPModel):
    """Class to perform a sequence of transformations.

    Attributes:
      transforms: list of transformation objects.

    Example::

      pipeline = TransformList([MVN(), PCA(pca_dim=128)], name="frontend")
    """

    def __init__(
        self, transforms: Union[HyperNPModel, List[HyperNPModel]], **kwargs: Any
    ) -> None:
        """Initializes a transformation pipeline.

        Args:
          transforms: Single transform or ordered list of transforms.
          **kwargs: Additional arguments forwarded to :class:`HyperNPModel`.
        """
        super().__init__(**kwargs)
        if transforms is None:
            raise ValueError("transforms cannot be None")
        if not isinstance(transforms, list):
            transforms = [transforms]
        for i, t in enumerate(transforms):
            if t is None:
                raise ValueError(f"transforms[{i}] cannot be None")
            if not isinstance(t, HyperNPModel):
                raise TypeError(
                    "Each transform must be a HyperNPModel instance, "
                    f"got {type(t).__name__} at index {i}"
                )
        self.transforms = transforms
        if transforms is not None:
            self.update_names()

    def _ensure_unique_transform_names(self) -> None:
        """Ensures each child transform has a unique name."""
        used = set()
        for i, t in enumerate(self.transforms):
            base_name = t.name if t.name is not None else f"transform-{i}"
            cand_name = base_name
            suffix = 1
            while cand_name in used:
                cand_name = f"{base_name}-{suffix}"
                suffix += 1
            if cand_name != t.name:
                t.name = cand_name
            used.add(cand_name)

    def append(self, t: HyperNPModel) -> None:
        """Appends a transformation to the list.

        Args:
          t: transformation object.
        """
        if t is None:
            raise ValueError("t cannot be None")
        if not isinstance(t, HyperNPModel):
            raise TypeError(
                "t must be a HyperNPModel instance, "
                f"got {type(t).__name__}"
            )
        self.transforms.append(t)
        self.update_names()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies the list of transformations to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies the list of transformations to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Applies the list of transformations to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        for t in self.transforms:
            x = t.predict(x)
        return x

    def update_names(self) -> None:
        """Prefixes child transform names with this pipeline name."""
        if self.name is not None:
            prefix = self.name + "/"
            for t in self.transforms:
                if t.name is None:
                    t.name = t.__class__.__name__
                if not t.name.startswith(prefix):
                    t.name = prefix + t.name
        self._ensure_unique_transform_names()

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict for the full pipeline."""
        config = super().get_config()
        config_t = {}
        for i in range(len(self.transforms)):
            config_t[str(i)] = self.transforms[i].get_config()
        config["transforms"] = config_t
        return config

    def save_params(self, f: h5py.File) -> None:
        """Saves all child transform parameters to the same HDF5 file.

        Args:
          f: Output HDF5 file handle.
        """
        names = [t.name for t in self.transforms]
        if len(names) != len(set(names)):
            raise ValueError(
                "Transform names must be unique to save parameters. "
                f"Got duplicate names: {names}"
            )
        for t in self.transforms:
            t.save_params(f)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "TransformList":
        """Loads a transformation pipeline from config and file parameters.

        Args:
          f: Input HDF5 file handle.
          config: Pipeline configuration dictionary.

        Returns:
          Loaded :class:`TransformList` instance.
        """
        config_ts = config["transforms"]
        transforms = []
        for i in range(len(config_ts)):
            config_t = config_ts.get(str(i), config_ts.get(i))
            if config_t is None:
                raise ValueError(
                    f"Missing transform config entry for index {i} in transforms."
                )
            logging.debug(config_t)
            class_name = config_t["class_name"]
            class_t = HyperNPModel.registry.get(class_name, globals().get(class_name))
            if class_t is None:
                raise ValueError(
                    f"Unknown transform class '{class_name}'. "
                    "Ensure the transform module is imported before loading."
                )
            t = class_t.load_params(f, config_t)
            transforms.append(t)
        return cls(transforms, name=config["name"])
