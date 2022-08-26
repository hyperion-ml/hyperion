"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from collections import OrderedDict as ODict
import re

import torch
from hyperion.torch.utils import dinossl

from .narchs import *
from .models import *


class TorchModelLoader(object):
    @staticmethod
    def _fix_compatibility(class_obj, cfg):
        """Function that fixed compatibility issues with deprecated models

        Args:
          class_obj: class type of the model.
          cfg: configuration dictiory that inits the model.

        Returns:
          Fixed configuration dictionary.
        """
        # for compatibility with older x-vector models
        if issubclass(class_obj, XVector):
            # We renamed AM-softmax scale parameer s to cos_scale
            if "s" in cfg:
                cfg["cos_scale"] = cfg["s"]
                del cfg["s"]

        return cfg

    @staticmethod
    def load(file_path, extra_objs={}, map_location=None, state_dict_key='model_state_dict', dinossl_kwargs=None):
        """
        Args:
            state_dict_key (str): key for state_dict of a pre-trained model. Currently either
                                'model_state_dict' or 'model_teacher_state_dict' (possible option in dinossl)
            dinossl_kwargs (dict): DINOHead related arguments to reconstruct the DINOHead module as it was in the traiing + location info for xvector extraction.
        """

        if map_location is None:
            map_location = torch.device("cpu")

        model_data = torch.load(file_path, map_location=map_location)
        cfg = model_data["model_cfg"]
        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name in globals():
            class_obj = globals()[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception("unknown object with class_name=%s" % (class_name))

        state_dict = model_data[state_dict_key]
        logging.info('Using state_dict_key: {} of the pre-trained model'.format(state_dict_key))

        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        cfg = TorchModelLoader._fix_compatibility(class_obj, cfg)

        p = re.compile("^module\.")
        q = re.compile('^backbone\.') # for dinossl
        num_tries = 3
        for tries in range(num_tries):
            try:
                model = class_obj.load(cfg=cfg, state_dict=state_dict)
                if (dinossl_kwargs is not None) and (dinossl_kwargs['dinossl_xvec_loc'] != 'f'): # no need when dinossl_kwargs['dinossl_xvec_loc'] == 'f' since it does not requires DINOHead
                    embed_dim = state_dict_head['mlp.0.weight'].shape[1]
                    model = dinossl.MultiCropWrapper(model, dinossl.DINOHead(embed_dim, dinossl_kwargs['dinossl_out_dim'], use_bn=dinossl_kwargs['dinossl_use_bn_in_head'],
                                        norm_last_layer=dinossl_kwargs['dinossl_norm_last_layer'], nlayers=dinossl_kwargs['dinossl_nlayers']))
                    model.head.load_state_dict(state_dict_head) # putting this into this "try:" block assumes the pre-trained model is always trained with multi-gpus.
                    model.dinossl_xvec_loc = dinossl_kwargs['dinossl_xvec_loc']
                return model
            except RuntimeError as err:
                # remove module prefix when is trained with dataparallel
                if tries == num_tries - 1:
                    # if it failed the 3 trials raise exception
                    raise err
                # remove module prefix when is trained with dataparallel
                state_dict = ODict((p.sub("", k), v) for k, v in state_dict.items())
                # below three are for dinossl
                state_dict = ODict((q.sub('',k), v) for k,v in state_dict.items())
                state_dict_head = ODict((k[5:], v) for k,v in state_dict.items() if (k[:4] == 'head'))
                state_dict = ODict((k, v) for k,v in state_dict.items() if not (k[:4] == 'head'))

