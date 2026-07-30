"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .audio_feats_mvn import AudioFeatsMVN
from .classif_head import ClassifHead
from .conformer_encoder_v1 import ConformerEncoderV1
from .convnext1d_encoder import ConvNext1dEncoder
from .convnext2d_encoder import ConvNext2dEncoder
from .dac_decoder import DACDecoder
from .dac_encoder import DACEncoder
from .dc1d_decoder import DC1dDecoder
from .dc1d_encoder import DC1dEncoder
from .dc2d_decoder import DC2dDecoder
from .dc2d_encoder import DC2dEncoder
from .dino_head import DINOHead
from .efficient_net import EfficientNet
from .etdnn import ETDNNV1
from .fcnet import FCNetV1, FCNetV2
from .feat_fuser_mvn import FeatFuserMVN
from .hydra_head_factory import HydraHeadFactory
from .hydra_heads import (
    HydraClassifHead,
    HydraClassifHeadOutput,
    HydraClassifLossType,
    HydraHead,
    HydraHeadType,
    HydraRegressionHeadOutput,
)
from .proj_head import ProjHead
from .qformer_v2 import QFormerV2
from .qproj_head import QProjHead
from .resetdnn import ResETDNNV1
from .resnet import (
    CFwSEIdRndResNet100,
    CFwSEIdRndResNet202,
    CFwSELRes2Net50,
    CFwSELRes2Next50_4x4d,
    CFwSELResNet18,
    CFwSELResNet34,
    CFwSELResNet50,
    CFwSELResNext50_4x4d,
    CFwSERes2Net18,
    CFwSERes2Net34,
    CFwSERes2Net50,
    CFwSERes2Net101,
    CFwSERes2Net152,
    CFwSERes2Next50_32x4d,
    CFwSERes2Next101_32x8d,
    CFwSEResNet18,
    CFwSEResNet34,
    CFwSEResNet50,
    CFwSEResNet101,
    CFwSEResNet152,
    CFwSEResNext50_32x4d,
    CFwSEResNext101_32x8d,
    CFwSEWideRes2Net50,
    CFwSEWideRes2Net101,
    CFwSEWideResNet50,
    CFwSEWideResNet101,
    FwSEIdRndResNet100,
    FwSEIdRndResNet202,
    FwSELRes2Net50,
    FwSELRes2Next50_4x4d,
    FwSELResNet18,
    FwSELResNet34,
    FwSELResNet50,
    FwSELResNext50_4x4d,
    FwSERes2Net18,
    FwSERes2Net34,
    FwSERes2Net50,
    FwSERes2Net101,
    FwSERes2Net152,
    FwSERes2Next50_32x4d,
    FwSERes2Next101_32x8d,
    FwSEResNet18,
    FwSEResNet34,
    FwSEResNet50,
    FwSEResNet101,
    FwSEResNet152,
    FwSEResNext50_32x4d,
    FwSEResNext101_32x8d,
    FwSEWideRes2Net50,
    FwSEWideRes2Net101,
    FwSEWideResNet50,
    FwSEWideResNet101,
    IdRndResNet100,
    IdRndResNet202,
    LRes2Net50,
    LRes2Next50_4x4d,
    LResNet18,
    LResNet34,
    LResNet34_345,
    LResNet50,
    LResNext50_4x4d,
    Res2Net18,
    Res2Net34,
    Res2Net50,
    Res2Net101,
    Res2Net152,
    Res2Next50_32x4d,
    Res2Next101_32x8d,
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNext50_32x4d,
    ResNext101_32x8d,
    SELRes2Net50,
    SELRes2Next50_4x4d,
    SELResNet18,
    SELResNet34,
    SELResNet50,
    SELResNext50_4x4d,
    SERes2Net18,
    SERes2Net34,
    SERes2Net50,
    SERes2Net101,
    SERes2Net152,
    SERes2Next50_32x4d,
    SERes2Next101_32x8d,
    SEResNet18,
    SEResNet34,
    SEResNet50,
    SEResNet101,
    SEResNet152,
    SEResNext50_32x4d,
    SEResNext101_32x8d,
    SEWideRes2Net50,
    SEWideRes2Net101,
    SEWideResNet50,
    SEWideResNet101,
    TSELRes2Net50,
    TSELRes2Next50_4x4d,
    TSELResNet18,
    TSELResNet34,
    TSELResNet50,
    TSELResNext50_4x4d,
    TSERes2Net18,
    TSERes2Net34,
    TSERes2Net50,
    TSERes2Net101,
    TSERes2Net152,
    TSERes2Next50_32x4d,
    TSERes2Next101_32x8d,
    TSEResNet18,
    TSEResNet34,
    TSEResNet50,
    TSEResNet101,
    TSEResNet152,
    TSEResNext50_32x4d,
    TSEResNext101_32x8d,
    TSEWideRes2Net50,
    TSEWideRes2Net101,
    TSEWideResNet50,
    TSEWideResNet101,
    WideRes2Net50,
    WideRes2Net101,
    WideResNet50,
    WideResNet101,
)
from .resnet1d_decoder import ResNet1dDecoder
from .resnet1d_encoder import ResNet1dEncoder
from .resnet2d_decoder import ResNet2dDecoder
from .resnet2d_encoder import ResNet2dEncoder
from .resnet_factory import ResNetFactory
from .rnn_encoder import RNNEncoder
from .rnn_transducer_decoder import RNNTransducerDecoder
from .spinenet import (
    LR0_SP53,
    R0_SP53,
    LSpine2Net49,
    LSpineNet49,
    LSpineNet49_5,
    LSpineNet49_bilinear,
    LSpineNet49_subpixel,
    SELSpine2Net49,
    SESpine2Net49,
    SESpine2Net49S,
    Spine2Net49,
    Spine2Net49S,
    SpineNet,
    SpineNet49,
    SpineNet49_concat_time,
    SpineNet49S,
    SpineNet96,
    SpineNet143,
    SpineNet190,
    TSELSpine2Net49,
    TSESpine2Net49,
    TSESpine2Net49S,
)
from .spinenet_factory import SpineNetFactory
from .streaming_dac_decoder import StreamingDACDecoder, StreamingDACDecoderState
from .streaming_dac_encoder import StreamingDACEncoder, StreamingDACEncoderState
from .tdnn import TDNNV1
from .tdnn_factory import TDNNFactory
from .torch_na_loader import TorchNALoader
from .transformer_encoder_v1 import TransformerEncoderV1
