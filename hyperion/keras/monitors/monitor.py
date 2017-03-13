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
    def __init__(self, file_path, verbose):
        super(Monitor, self).__init__()
        self.file_path = file_path
        self.verbose = verbose
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
    
    


# class DPTMonitor(Monitor):

#     def __init__(self, file_path, n_layers, verbose=1):
#         super(DPTMonitor, self).__init__(file_path,verbose)
#         self.n_layers=n_layers

#     def on_epoch_begin(self, epoch, logs={}):
#         self.t_start=time.time()

#     def on_epoch_end(self, epoch, logs={}):
#         epoch+=self.init_epoch
#         info_str=self.get_info_str(epoch,logs)
#         if self.verbose>0:
#             print(info_str)
#             sys.stdout.flush()
#         self.save_last(epoch,info_str)

#     def get_info_str(self,epoch,logs):
#         loss=logs.get('loss')
#         acc=-1.0
#         if 'acc' in logs:
#             acc=logs.get('acc')
#         elif 'masked_binary_accuracy' in logs:
#             acc=logs.get('masked_binary_accuracy')

#         if acc>-1.0:
#             info=('layers: %02d epoch: %05d loss: %f acc: %f '
#                   'elapsed_time: %.2f secs.') % (
#                       self.n_layers,epoch,loss,acc,
#                       time.time()-self.t_start)
#         else:
#             info=('layers: %02d epoch: %05d loss: %f '
#                   'elapsed_time: %.2f secs.') % (
#                       self.n_layers,epoch,loss,time.time()-self.t_start)
#         return info

#     def save_state(self,name,epoch):
#         file_state='%s/nn_state.%s.pickle' % (self.file_path,name)
#         rng_state=self.rng.get_state()
#         iteration=0
#         if hasattr(self.model.optimizer,'iteration'):
#             iteration=K.get_value(self.model.optimizer.iteration)
#             print('Save Iteration %d' % iteration)

#         f=open(file_state,'wb')
#         pickle.dump([epoch,rng_state,iteration],f)
#         f.close()

#     def load_state(self,name):
#         file_state='%s/nn_state.%s.pickle' % (self.file_path,name)
#         f=open(file_state,'rb')
#         try:
#             [epoch,rng_state,iteration]=pickle.load(f,encoding='latin1')
#         except TypeError:
#             [epoch,rng_state,iteration]=pickle.load(f)
#         f.close()
#         self.rng.set_state(rng_state)
#         self.init_epoch=epoch+1
#         self.iteration=iteration
#         return 

