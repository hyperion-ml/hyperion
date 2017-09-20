from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from ..hyp_model import HypModel
from .vae import *


class KerasModelLoader(object):

    @staticmethod
    def get_object():
        obj_dict={ 'VAE': VAE,
                   'CVAE': CVAE,
                   'TiedVAE_qYqZgY': TiedVAE_qYqZgY,
                   'TiedCVAE_qYqZgY': TiedCVAE_qYqZgY}
        return obj_dict
        
    
    @staticmethod
    def load(file_path):
        class_name = HypModel.load_config(file_path).class_name
        class_obj = KerasModelLoader.get_object()[class_name]
        return class_obj.load(file_path)

    
        
