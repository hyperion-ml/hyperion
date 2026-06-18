"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

import torch
import torch.nn as nn

from ..layer_blocks import (
    DC2dDecBlock,
    ResNet2dBasicDecBlock,
    ResNet2dBNDecBlock,
    SEResNet2dBasicDecBlock,
    SEResNet2dBNDecBlock,
)
from ..layers import ActivationFactory as AF
from ..layers import ICNR2d
from ..layers import NormLayer2dFactory as NLF
from ..layers import SubPixelConv2d
from .net_arch import NetArch


class ResNet2dDecoder(NetArch):
    """2D ResNet decoder.

    Attributes:
        in_channels (int): Number of channels in the decoder input.
        in_conv_channels (int): Channels in the stem convolution block.
        in_kernel_size (int): Kernel size of the stem convolution.
        in_stride (int): Stride of the stem convolution.
        resb_type (str): Residual block family.
        resb_repeats (List[int]): Number of residual blocks per stage.
        resb_channels (List[int]): Output channels per stage.
        resb_kernel_sizes (List[int]): Kernel sizes per stage.
        resb_strides (List[int]): Strides per stage.
        resb_dilations (List[int]): Dilations per stage.
        resb_groups (int): Number of convolution groups in residual blocks.
        head_channels (int): Optional output channels in the head block.
        hid_act (Any): Hidden activation specification.
        head_act (Any): Head activation specification.
        dropout_rate (float): Dropout probability.
        use_norm (bool): Whether normalization layers are enabled.
        norm_layer (Optional[str]): Normalization layer factory name.
        norm_before (bool): Whether normalization precedes activation.
        se_r (int): Squeeze-excitation reduction ratio.
    """

    def __init__(
        self,
        in_channels: int = 512,
        in_conv_channels: int = 512,
        in_kernel_size: int = 3,
        in_stride: int = 1,
        resb_type: str = "basic",
        resb_repeats: List[int] = [2, 2, 2, 2],
        resb_channels: Union[int, Sequence[int]] = [512, 256, 128, 64],
        resb_kernel_sizes: Union[int, Sequence[int]] = 3,
        resb_strides: Union[int, Sequence[int]] = 2,
        resb_dilations: Union[int, Sequence[int]] = 1,
        resb_groups: int = 1,
        head_channels: int = 0,
        hid_act: Union[str, Dict[str, Any]] = "relu",
        head_act: Optional[Union[str, Dict[str, Any]]] = None,
        dropout_rate: float = 0,
        se_r: int = 16,
        use_norm: bool = True,
        norm_layer: Optional[str] = None,
        norm_before: bool = True,
    ) -> None:
        """Build the decoder.

        Args:
            in_channels: Input channel dimension.
            in_conv_channels: Channels in the stem block.
            in_kernel_size: Kernel size of the stem block.
            in_stride: Stride of the stem block.
            resb_type: Residual block variant.
            resb_repeats: Number of blocks in each stage.
            resb_channels: Output channels for each stage.
            resb_kernel_sizes: Kernel sizes for each stage.
            resb_strides: Strides for each stage.
            resb_dilations: Dilations for each stage.
            resb_groups: Number of convolution groups in residual blocks.
            head_channels: Channels in the optional head block.
            hid_act: Hidden activation specification.
            head_act: Head activation specification.
            dropout_rate: Dropout probability.
            se_r: Squeeze-excitation reduction ratio.
            use_norm: Whether to use normalization layers.
            norm_layer: Normalization layer factory name.
            norm_before: Whether normalization precedes activation.
        """

        super().__init__()

        self.resb_type = resb_type
        bargs = {}
        if resb_type == "basic":
            self._block = ResNet2dBasicDecBlock
        elif resb_type == "bn":
            self._block = ResNet2dBNDecBlock
        elif resb_type == "sebasic":
            self._block = SEResNet2dBasicDecBlock
            bargs["se_r"] = se_r
        elif resb_type == "sebn":
            self._block = SEResNet2dBNDecBlock
            bargs["se_r"] = se_r
        else:
            raise ValueError(f"unsupported resb_type={resb_type}")

        self.in_channels = in_channels
        self.in_conv_channels = in_conv_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        num_superblocks = len(resb_repeats)
        self.resb_repeats = resb_repeats
        self.resb_channels = self._standarize_resblocks_param(
            resb_channels, num_superblocks, "resb_channels"
        )
        self.resb_kernel_sizes = self._standarize_resblocks_param(
            resb_kernel_sizes, num_superblocks, "resb_kernel_sizes"
        )
        self.resb_strides = self._standarize_resblocks_param(
            resb_strides, num_superblocks, "resb_strides"
        )
        self.resb_dilations = self._standarize_resblocks_param(
            resb_dilations, num_superblocks, "resb_dilations"
        )
        self.resb_groups = resb_groups
        self.head_channels = head_channels
        self.hid_act = hid_act
        self.head_act = head_act
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.se_r = se_r

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(min(self.resb_channels) // 2, 32)
            norm_groups = max(norm_groups, resb_groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        # stem block
        self.in_block = DC2dDecBlock(
            in_channels,
            in_conv_channels,
            in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            dropout_rate=dropout_rate,
            use_norm=use_norm,
            norm_layer=self._norm_layer,
            norm_before=norm_before,
        )
        self._context = self.in_block.context
        self._upsample_factor = self.in_block.stride

        cur_in_channels = in_conv_channels

        # middle blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_superblocks):
            repeats_i = self.resb_repeats[i]
            channels_i = self.resb_channels[i]
            stride_i = self.resb_strides[i]
            kernel_size_i = self.resb_kernel_sizes[i]
            dilation_i = self.resb_dilations[i]
            block_i = self._block(
                cur_in_channels,
                channels_i,
                kernel_size_i,
                stride=stride_i,
                dilation=1,
                groups=self.resb_groups,
                activation=hid_act,
                dropout_rate=dropout_rate,
                use_norm=use_norm,
                norm_layer=self._norm_layer,
                norm_before=norm_before,
                **bargs
            )

            self.blocks.append(block_i)
            self._context += block_i.context * self._upsample_factor
            self._upsample_factor *= block_i.upsample_factor

            for j in range(repeats_i - 1):
                block_i = self._block(
                    channels_i,
                    channels_i,
                    kernel_size_i,
                    stride=1,
                    dilation=dilation_i,
                    groups=self.resb_groups,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    use_norm=use_norm,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                    **bargs
                )

                self.blocks.append(block_i)
                self._context += block_i.context * self._upsample_factor

            cur_in_channels = channels_i

        # head feature block
        if self.head_channels > 0:
            self.head_block = DC2dDecBlock(
                cur_in_channels,
                head_channels,
                kernel_size=1,
                stride=1,
                activation=head_act,
                use_norm=False,
                norm_before=norm_before,
            )

        self._init_weights(hid_act)

    def _init_weights(self, hid_act: Union[str, Dict[str, Any]]) -> None:
        """Initialize convolution and normalization weights.

        Args:
            hid_act: Hidden activation specification used to select the
                Kaiming initializer nonlinearity.
        """
        act_name = "relu"
        if isinstance(hid_act, str):
            act_name = hid_act
        if isinstance(hid_act, dict):
            act_name = hid_act["name"]
        if act_name in ["relu6", "swish"]:
            act_name = "relu"

        init_f1 = lambda x: nn.init.kaiming_normal_(
            x, mode="fan_out", nonlinearity=act_name
        )
        init_f2 = lambda x: nn.init.kaiming_normal_(
            x, mode="fan_out", nonlinearity="relu"
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try:
                    init_f1(m.weight)
                except:
                    init_f2(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # re-init subpixelconvs
        for m in self.modules():
            if isinstance(m, SubPixelConv2d):
                try:
                    ICNR2d(m.conv.weight, stride=m.stride, initializer=init_f1)
                except:
                    ICNR2d(m.conv.weight, stride=m.stride, initializer=init_f2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if isinstance(hid_act, str):
        #             act_name = hid_act
        #         if isinstance(hid_act, dict):
        #             act_name = hid_act['name']
        #         if act_name == 'swish':
        #             act_name = 'relu'
        #         try:
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=act_name)
        #         except:
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    @staticmethod
    def _standarize_resblocks_param(
        p: Union[int, List[int]], num_blocks: int, p_name: str
    ) -> List[int]:
        """Normalize a residual-stage parameter to a list.

        Args:
            p: Scalar or sequence value to normalize.
            num_blocks: Number of residual stages.
            p_name: Parameter name used in error messages.

        Returns:
            A list with one entry per residual stage.
        """
        if isinstance(p, int):
            p = [p] * num_blocks
        elif isinstance(p, list):
            if len(p) == 1:
                p = p * num_blocks

            assert len(p) == num_blocks, "len(%s)(%d)!=%d" % (
                p_name,
                len(p),
                num_blocks,
            )
        else:
            raise TypeError("wrong type for param {}={}".format(p_name, p))

        return p

    def _compute_out_size(self, in_size: int) -> int:
        """Compute the output spatial size for a given input size.

        Args:
            in_size: Input spatial size.

        Returns:
            Output spatial size after all decoder stages.
        """
        out_size = in_size * self.in_stride

        for stride in self.resb_strides:
            out_size *= stride

        return out_size

    def in_context(self) -> Tuple[int, int]:
        """Return the receptive-field context required by the decoder.

        Returns:
            A symmetric `(left, right)` context tuple.
        """
        in_context = int(math.ceil(self._context / self._upsample_factor))
        return (in_context, in_context)

    def in_shape(self) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
        """Return the expected input shape.

        Returns:
            The expected `(batch, channels, height, width)` shape.
        """
        return (None, self.in_channels, None, None)

    def out_shape(
        self, in_shape: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
        """Return the output shape for a given input shape.

        Args:
            in_shape: Optional `(batch, channels, height, width)` input shape.

        Returns:
            The output `(batch, channels, height, width)` shape.
        """

        out_channels = (
            self.head_channels if self.head_channels > 0 else self.resb_channels[-1]
        )
        if in_shape is None:
            return (None, out_channels, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = None
        else:
            H = self._compute_out_size(in_shape[2])

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])

        return (in_shape[0], out_channels, H, W)

    def _match_shape(
        self, x: torch.Tensor, target_shape: Sequence[Optional[int]]
    ) -> torch.Tensor:
        """Center-crop the decoder output to match a target shape.

        Args:
            x: Decoder output tensor.
            target_shape: Target tensor shape.

        Returns:
            The cropped tensor.
        """
        x_dim = x.dim()
        ddim = x_dim - len(target_shape)
        for i in range(2, x_dim):
            surplus = x.size(i) - target_shape[i - ddim]
            assert surplus >= 0
            if surplus > 0:
                x = torch.narrow(x, i, surplus // 2, target_shape[i - ddim])

        return x.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        target_shape: Optional[Sequence[Optional[int]]] = None,
    ) -> torch.Tensor:
        """Run the decoder forward pass.

        Args:
            x: Input tensor.
            target_shape: Optional target shape for center cropping.

        Returns:
            The decoded tensor.
        """

        x = self.in_block(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if self.head_channels > 0:
            x = self.head_block(x)

        if target_shape is not None:
            x = self._match_shape(x, target_shape)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return the serializable configuration.

        Args:
            no_class_name: If `True`, omit the class metadata from the base
                configuration.

        Returns:
            A dictionary with the decoder configuration.
        """

        head_act = self.head_act
        hid_act = self.hid_act

        config = {
            "in_channels": self.in_channels,
            "in_conv_channels": self.in_conv_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "resb_type": self.resb_type,
            "resb_repeats": self.resb_repeats,
            "resb_channels": self.resb_channels,
            "resb_kernel_sizes": self.resb_kernel_sizes,
            "resb_strides": self.resb_strides,
            "resb_dilations": self.resb_dilations,
            "resb_groups": self.resb_groups,
            "head_channels": self.head_channels,
            "dropout_rate": self.dropout_rate,
            "hid_act": hid_act,
            "head_act": head_act,
            "se_r": self.se_r,
            "use_norm": self.use_norm,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments accepted by the decoder constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            The subset accepted by :meth:`__init__`.
        """

        # if "wo_norm" in kwargs:
        #     kwargs["use_norm"] = not kwargs["wo_norm"]
        #     del kwargs["wo_norm"]

        # if "norm_after" in kwargs:
        #     kwargs["norm_before"] = not kwargs["norm_after"]
        #     del kwargs["norm_after"]

        valid_args = (
            "in_channels",
            "in_conv_channels",
            "in_kernel_size",
            "in_stride",
            "resb_type",
            "resb_repeats",
            "resb_channels",
            "resb_kernel_sizes",
            "resb_strides",
            "resb_dilations",
            "resb_groups",
            "head_channels",
            "se_r",
            "hid_act",
            "head_act",
            "dropout_rate",
            "use_norm",
            "norm_layer",
            "norm_before",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """Add command-line arguments for this decoder.

        Args:
            parser: Argument parser to extend.
            prefix: Optional argument namespace prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--in-channels", type=int, default=80, help=("input channels of decoder")
        )

        parser.add_argument(
            "--in-conv-channels",
            default=128,
            type=int,
            help=("number of output channels in input convolution"),
        )

        parser.add_argument(
            "--in-kernel-size",
            default=3,
            type=int,
            help=("kernel size of input convolution"),
        )

        parser.add_argument(
            "--in-stride", default=1, type=int, help=("stride of input convolution")
        )

        parser.add_argument(
            "--resb-type",
            default="basic",
            choices=["basic", "bn", "sebasic", "sebn"],
            help=("residual blocks type"),
        )

        parser.add_argument(
            "--resb-repeats",
            default=[1, 1, 1],
            type=int,
            nargs="+",
            help=("resb-blocks repeats in each encoder stage"),
        )

        parser.add_argument(
            "--resb-channels",
            default=[128, 64, 32],
            type=int,
            nargs="+",
            help=("resb-blocks channels for each stage"),
        )

        parser.add_argument(
            "--resb-kernel-sizes",
            default=3,
            nargs="+",
            type=int,
            help=("resb-blocks kernels for each encoder stage"),
        )

        parser.add_argument(
            "--resb-strides",
            default=2,
            nargs="+",
            type=int,
            help=("resb-blocks strides for each encoder stage"),
        )

        parser.add_argument(
            "--resb-dilations",
            default=[1],
            nargs="+",
            type=int,
            help=("resb-blocks dilations for each encoder stage"),
        )

        parser.add_argument(
            "--resb-groups",
            default=1,
            type=int,
            help=("resb-blocks groups in convolutions"),
        )

        parser.add_argument(
            "--head-channels",
            default=0,
            type=int,
            help=("channels in the last conv block of encoder"),
        )

        try:
            parser.add_argument("--hid-act", default="relu", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--head-act", default=None, help="activation in encoder head"
        )

        try:
            parser.add_argument(
                "--dropout-rate", default=0, type=float, help="dropout probability"
            )
        except:
            pass

        try:
            parser.add_argument(
                "--norm-layer",
                default=None,
                choices=[
                    "batch-norm",
                    "group-norm",
                    "instance-norm",
                    "instance-norm-affine",
                    "layer-norm",
                ],
                help="type of normalization layer",
            )
        except:
            pass

        # parser.add_argument(
        #     "--wo-norm",
        #     default=False,
        #     action="store_true",
        #     help="without batch normalization",
        # )

        # parser.add_argument(
        #     "--norm-after",
        #     default=False,
        #     action="store_true",
        #     help="batch normalizaton after activation",
        # )
        parser.add_argument(
            "--use-norm",
            default=True,
            action=ActionYesNo,
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-before",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton before activation",
        )

        parser.add_argument(
            "--se-r",
            default=16,
            type=int,
            help=("squeeze-excitation compression ratio"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='ResNet2d decoder options')

    add_argparse_args = add_class_args
