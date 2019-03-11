from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import os
import logging

from ..hyp_model import HypModel
from .vae import *
from .embed import *

class KerasModelLoader(object):

    @staticmethod
    def get_object():
        obj_dict={ 'VAE': None,
                   'CVAE': None,
                   'TiedDVAE_QY' : TiedDVAE_QY,
                   'TiedSupVAE_QYQZgY': TiedSupVAE_QYQZgY,
                   'TiedCVAE_qYqZgY': None,
                   'SeqEmbed': SeqEmbed,
                   'SeqEmbedLDE': SeqEmbedLDE,
                   'SeqQEmbed': SeqQEmbed}
                   #'SeqEmbedAtt': SeqEmbedAtt }
        return obj_dict
        

    
    @staticmethod
    def load(model_path):
        json_file = model_path + '.json'
        class_name = HypModel.load_config(json_file)['class_name']
        class_obj = KerasModelLoader.get_object()[class_name]
        logging.info('Load model %s:%s' % (class_name, model_path))
        return class_obj.load(model_path)


    
    @staticmethod
    def load_checkpoint(model_path, epochs):
        found = False
        cur_epoch = 0
        for epoch in xrange(epochs,-1,-1):
            json_file = '%s/model.%04d.json' % (model_path, epoch)
            if os.path.isfile(json_file):
                found = True
                cur_epoch = epoch
                break
            
        if not found:
            json_file = '%s/model.best.json' % (model_path)
            if os.path.isfile(json_file):
                found = True
                
        model = None
        if found:
            file_path = json_file.rstrip('.json')
            model = KerasModelLoader.load(file_path)
        
        return model, cur_epoch 
