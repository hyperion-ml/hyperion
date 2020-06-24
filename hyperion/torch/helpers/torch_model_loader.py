"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from collections import OrderedDict as ODict
import re

import torch

from ..narchs import *
from ..seq_embed import XVector, TDNNXVector, ResNetXVector, TransformerXVectorV1, EfficientNetXVector
from ..models import VAE, VQVAE

class TorchModelLoader(object):

    @staticmethod
    def load(file_path, extra_objs={}, map_location=None):

        if map_location is None:
            map_location=torch.device('cpu')

        model_data = torch.load(file_path, map_location=map_location)
        cfg = model_data['model_cfg']
        class_name = cfg['class_name']
        del cfg['class_name']
        if class_name in globals():
            class_obj = globals()[class_name]
        elif class_name in extra_objs:
            class_obs = extra_objs[class_name]
        else:
            raise Exception('unknown object with class_name=%s' % (class_name))

        state_dict = model_data['model_state_dict']
        
        #remove module prefix when is trained with dataparallel
        p = re.compile('^module\.')
        state_dict = ODict((p.sub('',k),v) for k,v in state_dict.items())

        return class_obj.load(cfg=cfg, state_dict=state_dict)

    
