"""
Classes to monitor the training
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from keras import backend as K
from keras.callbacks import Callback
from keras.models import model_from_json

import sys
import os
import time
import numpy as np

import json
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

class Monitor(Callback):
    def __init__(self, file_path):
        super(Monitor, self).__init__()
        self.file_path = file_path
        self.init_epoch = 0
        self.t_start = 0
        self.rng = None
        self.iteration = 0

    def on_train_begin(self, logs={}):
        if self.init_epoch > 0 and self.iteration > 0:
            K.set_value(self.model.optimizer.iteration, self.iteration)
        self.iteration = 0
        
    def on_epoch_begin(self, epoch, logs={}):
        self.t_start = time.time()

    def save_model(self,name, info_str):
        if not(os.path.isdir(self.file_path)):
            os.makedirs(self.file_path)
        file_model = '%s/nn.%s.h5' % (self.file_path, name)
        file_info = '%s/nn.%s.info' % (self.file_path, name)
        self.model.save(file_model)
        f = open(file_info,'w').write('%s\n' % info_str)

    def load_model(self,name):
        file_model = '%s/nn.%s.h5' % (self.file_path, name)
        return F.load_model(file_model)

    def save_best(self,info_str):
        self.save_model('best',info_str)

    def save_last(self,epoch,info_str):
        self.save_model('last',info_str)
        self.save_state('last',epoch)
        file_info = '%s/nn.%05d.info' % (self.file_path, epoch)
        f = open(file_info,'w').write('%s\n' % info_str)

    def load_last(self):
        model = self.load_model('last')
        if not(model is None):
            self.load_state('last')
        return model

    def set_rng(self,rng):
        self.rng=rng
    
    

