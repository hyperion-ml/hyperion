"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .activation_factory import ActivationFactory
from .attention import (
    LocalScaledDotProdAttRelPosEncV1,
    LocalScaledDotProdAttV1,
    ScaledDotProdAttRelPosEncV1,
    ScaledDotProdAttV1,
)
from .audio_feats import *
from .audio_feats_factory import AudioFeatsFactory
from .calibrators import LinBinCalibrator
from .dropout import DropConnect1d, DropConnect2d, Dropout1d, DropPath1d, DropPath2d
from .feat_fuser_factory import FeatFuserFactory
from .feat_fusers import (
    CatFeatFuser,
    LastFeatFuser,
    LinearFeatFuser,
    WeightedAvgFeatFuser,
)
from .global_pool import *
from .grn import GRN1d, GRN2d
from .interpolate import Interpolate
from .lora import LoRAFactory
from .margin_losses import ArcLossOutput, CosLossOutput, SubCenterArcLossOutput
from .mvn import MeanVarianceNorm
from .norm_layer_factory import NormLayer1dFactory, NormLayer2dFactory
from .norm_layers import RMSNorm
from .pool_factory import GlobalPool1dFactory
from .pos_encoder import (
    ConvPosEncoder,
    NoPosEncoder,
    PosEncoder,
    PosEncoderBase,
    RelPosEncoder,
)
from .spec_augment import AxisMasker, SpecAugment, SpecWarper
from .subpixel_convs import ICNR1d, ICNR2d, SubPixelConv1d, SubPixelConv2d
