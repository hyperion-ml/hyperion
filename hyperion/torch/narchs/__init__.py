"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .fcnet import FCNetV1
from .tdnn import TDNNV1
from .etdnn import ETDNNV1
from .resetdnn import ResETDNNV1
from .resnet import *
from .classif_head import ClassifHead


from .tdnn_factory import TDNNFactory
from .resnet_factory import ResNetFactory
