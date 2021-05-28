"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

# xvectors had been moved to models
# we import them here for backwards compatibility

from ..models.xvector import XVector
from ..models.tdnn_xvector import TDNNXVector
from ..models.resnet_xvector import ResNetXVector
from ..models.efficient_net_xvector import EfficientNetXVector
from ..models.transformer_xvector_v1 import TransformerXVectorV1
from ..models.spinenet_xvector import SpineNetXVector
