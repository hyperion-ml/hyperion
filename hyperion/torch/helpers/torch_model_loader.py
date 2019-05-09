"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch

from ..narchs import *
from ..seq_embed import XVector

class TorchModelLoader(object):

    @staticmethod
    def load(file_path, extra_objs={}):

        model_data = torch.load(model_path)
        cfg = model_data['model_cfg']
        class_name = cfg['class_name']
        del cfg['class_name']
        if class_name is in globals():
            class_obj = globals()[class_name]
        elif class_name is in extra_objs:
            class_obs = extra_objs[class_name]
        else:
            raise Exception('unknown object with class_name=%s' % (class_name))

        state_dict = model_data['model_state_dict']
        return class_obj.load(cfg=cfg, state_dict=state_dict)

    
