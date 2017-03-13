"""
List of transformations
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import h5py

from ..hyp_model import HypModel

from .cent_whiten import CentWhiten
from .lnorm import LNorm
from .pca import PCA
from .lda import LDA
from .nap import NAP
from .mvn import MVN


class TransformList(HypModel):
    
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
            t.name = self.name + '/' + t.name

            
    def predict(self, x):
        for t in self.transforms:
            x = t.predict(x)
        return x

    
    def update_names(self):
        if self.name is not None:
            for t in self.transforms:
                t.name = self.name + '/' + t.name

    
    def get_config(self):
        config = super(TransformList, self).get_config()
        config_t = {}
        for i in xrange(len(self.transforms)):
            config_t[i] = self.transforms[i].get_config()
        config['transforms'] = config_t
        return config

    
    def save_params(self, f):
        for t in self.transforms:
            t.save_params(f)


    @classmethod
    def load_params(cls, f, config):
        config_ts = config['transforms']
        transforms = []
        for i in xrange(len(config_ts)):
            config_t = config_ts[str(i)]
            print(i, config_t)
            class_t = globals()[config_t['class_name']]
            t = class_t.load_params(f, config_t)
            transforms.append(t)
        return cls(transforms, name=config['name'])
            
