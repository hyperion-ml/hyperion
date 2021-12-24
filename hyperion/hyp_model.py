"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from abc import ABCMeta, abstractmethod
import os
import json
from copy import deepcopy

import numpy as np
import h5py

from .hyp_defs import float_save, float_cpu


class HypModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None, **kwargs):
        self.name = name
        self._is_init = False

    def copy(self):
        return deepcopy(self)

    @property
    def is_init(self):
        return self._is_init

    def init_to_false(self):
        self._is_init = False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def fit(self, x, sample_weights=None, x_val=None, sample_weights_val=None):
        pass

    @abstractmethod
    def fit_generator(self, x, x_val=None):
        pass

    @abstractmethod
    def save(self, file_path):
        file_dir = os.path.dirname(file_path)
        if not (os.path.isdir(file_dir)):
            os.makedirs(file_dir, exist_ok=True)
        with h5py.File(file_path, "w") as f:
            config = self.to_json()
            f.create_dataset("config", data=np.array(config, dtype="S"))
            self.save_params(f)

    @abstractmethod
    def save_params(self, f):
        assert True, "save_params method not defined for %s" % (self.__class__.__name__)

    def _save_params_from_dict(self, f, params, dtypes=None):
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
        try:
            with h5py.File(file_path, "r") as f:
                json_str = str(np.asarray(f["config"]).astype("U"))
                return cls.load_config_from_json(json_str)
        except:
            with open(file_path, "r") as f:
                return cls.load_config_from_json(f.read())

    @classmethod
    def load(cls, file_path):
        with h5py.File(file_path, "r") as f:
            json_str = str(np.asarray(f["config"]).astype("U"))
            config = cls.load_config_from_json(json_str)
            return cls.load_params(f, config)

    @classmethod
    def load_params(cls, f, config):
        return cls(name=config["name"])

    @staticmethod
    def _load_params_to_dict(f, name, params, dtypes=None):
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

    @abstractmethod
    def get_config(self):
        config = {"class_name": self.__class__.__name__, "name": self.name}
        return config

    def to_json(self, **kwargs):
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
        return json.loads(json_str)
