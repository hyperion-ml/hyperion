"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from collections import OrderedDict as ODict
import re

import torch

from .narchs import *
from .models import *


class TorchModelLoader(object):
    @staticmethod
    def load(file_path, extra_objs={}, map_location=None):

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

        state_dict = model_data["model_state_dict"]

        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        # for compatibility with older x-vector models
        if isinstance(class_obj, XVector):
            # We renamed AM-softmax scale parameer s to cos_scale
            if "s" in cfg:
                cfg["cos_scale"] = cfg["s"]
                del cfg["s"]

        p = re.compile("^module\.")
        num_tries = 3
        for tries in range(num_tries):
            try:
                return class_obj.load(cfg=cfg, state_dict=state_dict)
            except RuntimeError as err:
                # remove module prefix when is trained with dataparallel
                if tries == num_tries - 1:
                    # if it failed the 3 trials raise exception
                    raise err
                # remove module prefix when is trained with dataparallel
                state_dict = ODict((p.sub("", k), v) for k, v in state_dict.items())
