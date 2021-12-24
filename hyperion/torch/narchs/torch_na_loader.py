"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch

from .fcnet import FCNetV1

from .tdnn import TDNNV1
from .etdnn import ETDNNV1
from .resetdnn import ResETDNNV1

from .resnet import *

from .transformer_encoder_v1 import TransformerEncoderV1
from .conformer_encoder_v1 import ConformerEncoderV1

from .dc1d_encoder import DC1dEncoder
from .dc1d_decoder import DC1dDecoder
from .dc2d_encoder import DC2dEncoder
from .dc2d_decoder import DC2dDecoder

from .resnet1d_encoder import ResNet1dEncoder
from .resnet1d_decoder import ResNet1dDecoder
from .resnet2d_encoder import ResNet2dEncoder
from .resnet2d_decoder import ResNet2dDecoder

from .efficient_net import EfficientNet

from .classif_head import ClassifHead

from .audio_feats_mvn import AudioFeatsMVN


class TorchNALoader(object):
    @staticmethod
    def load(file_path, extra_objs={}):

        model_data = torch.load(model_path)
        cfg = model_data["model_cfg"]
        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name in globals():
            class_obj = globals()[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception(
                "unknown neural architecture object with class_name=%s" % (class_name)
            )

        state_dict = model_data["model_state_dict"]
        return class_obj.load(cfg=cfg, state_dict=state_dict)

    @staticmethod
    def load_from_cfg(cfg, state_dict=None, extra_objs={}):
        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name in globals():
            class_obj = globals()[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception(
                "unknown neural architecture object with class_name=%s" % (class_name)
            )

        return class_obj.load(cfg=cfg, state_dict=state_dict)
