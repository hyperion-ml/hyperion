"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import numpy as np
import h5py

from ..hyp_model import HypModel

from .cent_whiten import CentWhiten
from .cent_whiten_up import CentWhitenUP
from .lnorm import LNorm
from .lnorm_up import LNormUP
from .pca import PCA
from .lda import LDA
from .nda import NDA
from .nap import NAP
from .mvn import MVN
from .gaussianizer import Gaussianizer


class TransformList(HypModel):
    """Class to perform a list of transformations"""

    def __init__(self, transforms, **kwargs):
        super(TransformList, self).__init__(**kwargs)
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        if transforms is not None:
            self.update_names()

    def append(self, t):
        self.transforms.append(t)
        if self.name is not None:
            t.name = self.name + "/" + t.name

    def predict(self, x):
        for t in self.transforms:
            x = t.predict(x)
        return x

    def update_names(self):
        if self.name is not None:
            for t in self.transforms:
                t.name = self.name + "/" + t.name

    def get_config(self):
        config = super(TransformList, self).get_config()
        config_t = {}
        for i in range(len(self.transforms)):
            config_t[i] = self.transforms[i].get_config()
        config["transforms"] = config_t
        return config

    def save_params(self, f):
        for t in self.transforms:
            t.save_params(f)

    @classmethod
    def load_params(cls, f, config):
        config_ts = config["transforms"]
        transforms = []
        for i in range(len(config_ts)):
            config_t = config_ts[str(i)]
            logging.debug(config_t)
            class_t = globals()[config_t["class_name"]]
            t = class_t.load_params(f, config_t)
            transforms.append(t)
        return cls(transforms, name=config["name"])
