"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import json
from copy import deepcopy

import numpy as np
import h5py

from ..hyp_defs import float_save, float_cpu


class NPModel(object):
    """Base class for machine learning models based on numpy.

    Attributes:
      name: optional identifier for the model.
    """

    def __init__(self, name=None, **kwargs):
        self.name = name
        self._is_init = False

    def copy(self):
        """Returns a clone of the model."""
        return deepcopy(self)

    def clone(self):
        """Returns a clone of the model."""
        return deepcopy(self)

    @property
    def is_init(self):
        """Returns True if the model has been initialized."""
        return self._is_init

    def init_to_false(self):
        """Sets the model as non initialized."""
        self._is_init = False

    def initialize(self):
        pass

    def fit(self, x, sample_weight=None, x_val=None, sample_weight_val=None):
        """Trains the model.

        Args:
          x: train data matrix with shape (num_samples, x_dim).
          sample_weight: weight of each sample in the training loss shape (num_samples,).
          x_val: validation data matrix with shape (num_val_samples, x_dim).
          sample_weight_val: weight of each sample in the val. loss.
        """
        raise NotImplementedError()

    def fit_generator(self, x, x_val=None):
        """Trains the model from a data generator function.

        Args:
          x: train data generation function.
          x_val: validation data generation function.
        """
        raise NotImplementedError()

    def save(self, file_path):
        """Saves the model to file.

        Args:
          file_path: filename path.
        """
        file_dir = os.path.dirname(file_path)
        if not (os.path.isdir(file_dir)):
            os.makedirs(file_dir, exist_ok=True)
        with h5py.File(file_path, "w") as f:
            config = self.to_json()
            f.create_dataset("config", data=np.array(config, dtype="S"))
            self.save_params(f)

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        raise NotImplementedError(
            f"save_params method not defined for {self.__class__.__name__}"
        )

    def _save_params_from_dict(self, f, params, dtypes=None):
        """Saves a dictionary of model parameters into the file.

        Args:
          f: file handle.
          params: dictionary of model parameters.
          dtypes: dictionary indicating the dtypes of the model parameters.
        """
        if dtypes is None:
            dtypes = dict((k, float_save()) for k in params)

        if self.name is None:
            prefix = ""
        else:
            prefix = self.name + "/"
        for k, v in params.items():
            if v is None:
                continue
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            p_name = prefix + k
            f.create_dataset(p_name, data=v.astype(dtypes[k], copy=False))

    @classmethod
    def load_config(cls, file_path):
        """Loads the model configuration from file.

        Args:
          file_path: path to the file where the model is stored.

        Returns:
          Dictionary containing the model configuration.
        """
        try:
            with h5py.File(file_path, "r") as f:
                json_str = str(np.asarray(f["config"]).astype("U"))
                return cls.load_config_from_json(json_str)
        except:
            with open(file_path, "r") as f:
                return cls.load_config_from_json(f.read())

    @classmethod
    def load(cls, file_path):
        """Loads the model from file.

        Args:
          file_path: path to the file where the model is stored.

        Returns:
          Model object.
        """
        with h5py.File(file_path, "r") as f:
            json_str = str(np.asarray(f["config"]).astype("U"))
            config = cls.load_config_from_json(json_str)
            return cls.load_params(f, config)

    @classmethod
    def load_params(cls, f, config):
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        return cls(name=config["name"])

    @staticmethod
    def _load_params_to_dict(f, name, params, dtypes=None):
        """Loads the model parameters from file to a dictionary.

        Args:
          f: file handle.
          name: model identifier or None.
          params: parameter names.
          dtypes: dictionary containing the dtypes of the parameters.

        Returns:
          Dictionary with model parameters.
        """
        if dtypes is None:
            dtypes = dict((k, float_cpu()) for k in params)
        if name is None:
            prefix = ""
        else:
            prefix = name + "/"

        param_dict = {}
        for k in params:
            p_name = prefix + k
            if p_name in f:
                param_dict[k] = np.asarray(f[p_name]).astype(
                    dtype=dtypes[k], copy=False
                )
            else:
                param_dict[k] = None
        return param_dict

    def get_config(self):
        """Returns the model configuration dict."""
        config = {"class_name": self.__class__.__name__, "name": self.name}
        return config

    def to_json(self, **kwargs):
        """Returns model config as json string."""
        # Piece of code borrowed from keras
        def get_json_type(obj):
            # if obj is any numpy type
            if type(obj).__module__ == np.__name__:
                return obj.item()

            # if obj is a python 'type'
            if type(obj).__name__ == type.__name__:
                return obj.__name__

            raise TypeError("Not JSON Serializable:", obj)

        config = self.get_config()
        return json.dumps(config, default=get_json_type, **kwargs)

    @staticmethod
    def load_config_from_json(json_str):
        """Converts json string into dict."""
        return json.loads(json_str)
