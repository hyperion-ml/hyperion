"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

import torch
import torch.nn as nn

from ..layer_blocks import DC2dEncBlock
from ..layers import NormLayer2dFactory as NLF
from .net_arch import NetArch


class DC2dEncoder(NetArch):
    """2D convolutional encoder for spectrogram-like inputs.

    Attributes:
        in_channels: Number of input feature channels.
        in_conv_channels: Number of channels in the stem convolution.
        in_kernel_size: Kernel size of the stem convolution.
        in_stride: Stride of the stem convolution.
        conv_repeats: Number of blocks in each encoder stage.
        conv_channels: Output channels for each encoder stage.
        conv_kernel_sizes: Kernel size for each encoder stage.
        conv_strides: Stride for each encoder stage.
        conv_dilations: Dilation for blocks in each encoder stage.
        head_channels: Number of output channels in the optional head block.
        hid_act: Hidden activation specification.
        head_act: Activation specification for the head block.
        dropout_rate: Dropout probability used in the convolution blocks.
        use_norm: Whether normalization layers are enabled.
        norm_layer: Normalization layer type.
        norm_before: If True, normalization is applied before activation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        in_conv_channels: int = 128,
        in_kernel_size: int = 3,
        in_stride: int = 1,
        conv_repeats: Sequence[int] = [1, 1, 1],
        conv_channels: Union[int, Sequence[int]] = [128, 64, 32],
        conv_kernel_sizes: Union[int, Sequence[int]] = 3,
        conv_strides: Union[int, Sequence[int]] = 2,
        conv_dilations: Union[int, Sequence[int]] = 1,
        head_channels: int = 0,
        hid_act: Any = "relu",
        head_act: Optional[Any] = None,
        dropout_rate: float = 0,
        use_norm: bool = True,
        norm_layer: Optional[str] = None,
        norm_before: bool = True,
    ) -> None:
        """Initialize a 2D convolutional encoder.

        Args:
            in_channels: Number of input feature channels.
            in_conv_channels: Number of channels in the stem convolution.
            in_kernel_size: Kernel size of the stem convolution.
            in_stride: Stride of the stem convolution.
            conv_repeats: Number of blocks in each encoder stage.
            conv_channels: Output channels for each encoder stage.
            conv_kernel_sizes: Kernel size for each encoder stage.
            conv_strides: Stride for each encoder stage.
            conv_dilations: Dilation for blocks in each encoder stage.
            head_channels: Number of output channels in the head block.
            hid_act: Hidden activation specification.
            head_act: Activation specification for the head block.
            dropout_rate: Dropout probability used in the convolution blocks.
            use_norm: Whether to enable normalization layers.
            norm_layer: Normalization layer type.
            norm_before: If True, apply normalization before activation.
        """

        super().__init__()
        self.in_channels = in_channels
        self.in_conv_channels = in_conv_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        num_superblocks = len(conv_repeats)
        self.conv_repeats = conv_repeats
        self.conv_channels = self._standarize_convblocks_param(
            conv_channels, num_superblocks, "conv_channels"
        )
        self.conv_kernel_sizes = self._standarize_convblocks_param(
            conv_kernel_sizes, num_superblocks, "conv_kernel_sizes"
        )
        self.conv_strides = self._standarize_convblocks_param(
            conv_strides, num_superblocks, "conv_strides"
        )
        self.conv_dilations = self._standarize_convblocks_param(
            conv_dilations, num_superblocks, "conv_dilations"
        )
        self.head_channels = head_channels
        self.hid_act = hid_act
        self.head_act = head_act
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_before = norm_before

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(min(self.conv_channels) // 2, 32)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        # stem block
        self.in_block = DC2dEncBlock(
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
        self._downsample_factor = self.in_block.stride

        cur_in_channels = in_conv_channels

        # middle blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_superblocks):
            repeats_i = self.conv_repeats[i]
            channels_i = self.conv_channels[i]
            stride_i = self.conv_strides[i]
            kernel_size_i = self.conv_kernel_sizes[i]
            dilation_i = self.conv_dilations[i]
            block_i = DC2dEncBlock(
                cur_in_channels,
                channels_i,
                kernel_size_i,
                stride=stride_i,
                dilation=dilation_i,
                activation=hid_act,
                dropout_rate=dropout_rate,
                use_norm=use_norm,
                norm_layer=self._norm_layer,
                norm_before=norm_before,
            )

            self.blocks.append(block_i)
            self._context += block_i.context * self._downsample_factor
            self._downsample_factor *= block_i.stride

            for j in range(repeats_i - 1):
                block_i = DC2dEncBlock(
                    channels_i,
                    channels_i,
                    kernel_size_i,
                    stride=1,
                    dilation=dilation_i,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    use_norm=use_norm,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                )

                self.blocks.append(block_i)
                self._context += block_i.context * self._downsample_factor

            cur_in_channels = channels_i

        # head feature block
        if self.head_channels > 0:
            self.head_block = DC2dEncBlock(
                cur_in_channels,
                head_channels,
                kernel_size=1,
                stride=1,
                activation=head_act,
                use_norm=False,
                norm_before=norm_before,
            )

        self._init_weights(hid_act)

    def _init_weights(self, hid_act: Any) -> None:
        """Initialize convolution and batch-normalization parameters.

        Args:
            hid_act: Hidden activation specification used to choose the
                Kaiming initialization nonlinearity.
        """
        act_name = "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act["name"]
                if act_name == "swish":
                    act_name = "relu"
                try:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=act_name
                    )
                except:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _standarize_convblocks_param(
        p: Union[int, Sequence[int]], num_blocks: int, p_name: str
    ) -> Sequence[int]:
        """Normalize a per-block argument to one value per encoder stage.

        Args:
            p: Scalar or sequence parameter to normalize.
            num_blocks: Number of encoder stages expected.
            p_name: Parameter name used in error messages.

        Returns:
            Sequence[int]: Parameter values expanded to `num_blocks`.
        """
        if isinstance(p, int):
            p = [p] * num_blocks
        elif isinstance(p, (list, tuple)):
            if len(p) == 1:
                p = list(p) * num_blocks

            assert len(p) == num_blocks, "len(%s)(%d)!=%d" % (
                p_name,
                len(p),
                num_blocks,
            )
        else:
            raise TypeError("wrong type for param {}={}".format(p_name, p))

        return p

    def _compute_out_size(self, in_size: int) -> int:
        """Compute the output spatial size after all strided stages.

        Args:
            in_size: Input spatial size.

        Returns:
            int: Output spatial size.
        """
        out_size = int((in_size - 1) // self.in_stride + 1)

        for stride in self.conv_strides:
            out_size = int((out_size - 1) // stride + 1)

        return out_size

    def in_context(self) -> Tuple[int, int]:
        """Return the left and right spatial context.

        Returns:
            Tuple[int, int]: Symmetric spatial context in pixels.
        """
        return (self._context, self._context)

    def in_shape(self) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
        """Return the expected input shape.

        Returns:
            Tuple[Optional[int], int, Optional[int], Optional[int]]: Batch,
            channel, height, and width dimensions.
        """
        return (None, self.in_channels, None, None)

    def out_shape(
        self, in_shape: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
        """Return the output shape for an optional input shape.

        Args:
            in_shape: Optional input shape used to infer the output height and
                width.

        Returns:
            Tuple[Optional[int], int, Optional[int], Optional[int]]: Output
            shape.
        """

        out_channels = (
            self.head_channels if self.head_channels > 0 else self.conv_channels[-1]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder on an input tensor.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Encoded tensor.
        """

        x = self.in_block(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if self.head_channels > 0:
            x = self.head_block(x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return the configuration needed to recreate the module.

        Args:
            no_class_name: If True, omit the class name from the base config.

        Returns:
            Dict[str, Any]: Serializable configuration dictionary.
        """

        head_act = self.head_act
        hid_act = self.hid_act

        config = {
            "in_channels": self.in_channels,
            "in_conv_channels": self.in_conv_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "conv_repeats": self.conv_repeats,
            "conv_channels": self.conv_channels,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "conv_strides": self.conv_strides,
            "conv_dilations": self.conv_dilations,
            "head_channels": self.head_channels,
            "dropout_rate": self.dropout_rate,
            "hid_act": hid_act,
            "head_act": head_act,
            "use_norm": self.use_norm,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments to those handled by the constructor.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by the constructor.
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
            "conv_repeats",
            "conv_channels",
            "conv_kernel_sizes",
            "conv_strides",
            "conv_dilations",
            "head_channels",
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
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        head_channels: bool = False,
    ) -> None:
        """Add encoder arguments to an argument parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix used to create a nested parser entry.
            head_channels: If True, expose the ``head_channels`` argument.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--in-channels", type=int, default=1, help=("input channel dimension")
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
            "--conv-repeats",
            default=[1, 1, 1],
            type=int,
            nargs="+",
            help=("conv-blocks repeats in each encoder stage"),
        )

        parser.add_argument(
            "--conv-channels",
            default=[128, 64, 32],
            type=int,
            nargs="+",
            help=("conv-blocks channels for each stage"),
        )

        parser.add_argument(
            "--conv-kernel-sizes",
            default=[3],
            nargs="+",
            type=int,
            help=("conv-blocks kernels for each encoder stage"),
        )

        parser.add_argument(
            "--conv-strides",
            default=[2],
            nargs="+",
            type=int,
            help=("conv-blocks strides for each encoder stage"),
        )

        parser.add_argument(
            "--conv-dilations",
            default=[1],
            nargs="+",
            type=int,
            help=("conv-blocks dilations for each encoder stage"),
        )

        if head_channels:
            parser.add_argument(
                "--head-channels",
                default=16,
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

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='DC2d encoder options')

    add_argparse_args = add_class_args
