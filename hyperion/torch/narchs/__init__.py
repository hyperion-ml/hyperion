"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

from .fcnet import FCNetV1
from .tdnn import TDNNV1
from .etdnn import ETDNNV1
from .resetdnn import ResETDNNV1
from .resnet import *
from .transformer_encoder_v1 import TransformerEncoderV1
from .dc1d_encoder import DC1dEncoder
from .dc1d_decoder import DC1dDecoder
from .efficient_net import EfficientNet
from .classif_head import ClassifHead

from .tdnn_factory import TDNNFactory
from .resnet_factory import ResNetFactory
