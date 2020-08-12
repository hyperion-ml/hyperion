"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

from .dropout import Dropout1d, DropConnect2d
from .global_pool import *

from .activation_factory import ActivationFactory
from .norm_layer_factory import NormLayer2dFactory, NormLayer1dFactory
from .pool_factory import GlobalPool1dFactory

from .margin_losses import CosLossOutput, ArcLossOutput

from .audio_feats import *
from .audio_feats_factory import AudioFeatsFactory

from .mvn import MeanVarianceNorm

from .attention import ScaledDotProdAttV1, LocalScaledDotProdAttV1
from .pos_encoder import PosEncoder

from .subpixel_convs import SubPixelConv1d, SubPixelConv2d, ICNR1d, ICNR2d
