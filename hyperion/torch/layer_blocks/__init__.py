"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .fc_blocks import FCBlock
from .se_blocks import SEBlock2D, TSEBlock2D, SEBlock2d, TSEBlock2d, SEBlock1d
from .tdnn_blocks import TDNNBlock
from .etdnn_blocks import ETDNNBlock
from .resetdnn_blocks import ResETDNNBlock
from .resnet_blocks import ResNetInputBlock, ResNetBasicBlock, ResNetBNBlock
from .resnet_blocks import ResNetEndpointBlock
from .seresnet_blocks import SEResNetBasicBlock, SEResNetBNBlock
from .res2net_blocks import Res2NetBasicBlock, Res2NetBNBlock
from .mbconv_blocks import MBConvBlock, MBConvInOutBlock
from .transformer_feedforward import PositionwiseFeedForward, Conv1dx2, Conv1dLinear
from .transformer_encoder_v1 import TransformerEncoderBlockV1
from .transformer_conv2d_subsampler import TransformerConv2dSubsampler
from .conformer_conv import ConformerConvBlock
from .conformer_encoder_v1 import ConformerEncoderBlockV1
from .dc1d_blocks import DC1dEncBlock, DC1dDecBlock
from .dc2d_blocks import DC2dEncBlock, DC2dDecBlock
from .resnet1d_blocks import (
    ResNet1dBasicBlock,
    ResNet1dBasicDecBlock,
    ResNet1dBNBlock,
    ResNet1dBNDecBlock,
    ResNet1dEndpoint,
)
from .resnet1d_blocks import (
    SEResNet1dBasicBlock,
    SEResNet1dBasicDecBlock,
    SEResNet1dBNBlock,
    SEResNet1dBNDecBlock,
)
from .res2net1d_blocks import Res2Net1dBasicBlock, Res2Net1dBNBlock
from .resnet2d_blocks import (
    ResNet2dBasicBlock,
    ResNet2dBasicDecBlock,
    ResNet2dBNBlock,
    ResNet2dBNDecBlock,
)
from .resnet2d_blocks import (
    SEResNet2dBasicBlock,
    SEResNet2dBasicDecBlock,
    SEResNet2dBNBlock,
    SEResNet2dBNDecBlock,
)
from .res2net2d_blocks import Res2Net2dBasicBlock, Res2Net2dBNBlock
from .spine_blocks import BlockSpec, SpineResample, SpineEndpoints, SpineConv
