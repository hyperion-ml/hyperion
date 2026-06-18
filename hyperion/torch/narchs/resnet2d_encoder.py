"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks import (
    DC2dEncBlock,
    Res2Net2dBasicBlock,
    Res2Net2dBNBlock,
    ResNet2dEndpoint,
    ResNet2dBasicBlock,
    ResNet2dBNBlock,
    SEResNet2dBasicBlock,
    SEResNet2dBNBlock,
)
from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


class ResNet2dEncoder(NetArch):
    """2D ResNet encoder.

    This variant exposes additional residual-block choices and optional
    multi-scale configuration compared with the simpler ResNet encoder.

    Attributes:
        in_channels (int): Input channel dimension.
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
        drop_connect_rate (float): Drop-connect probability.
        se_r (int): Squeeze-excitation reduction ratio.
        time_se (bool): Whether SE pooling is time-only.
        in_feats (Optional[int]): Input feature dimension for `time_se`.
        res2net_width_factor (float): Res2Net width scaling factor.
        res2net_scale (int): Res2Net scale factor.
        multilayer (bool): Enable multi-layer feature aggregation.
        multilayer_concat (bool): Concatenate instead of averaging endpoints.
        endpoint_channels (Optional[int]): Output channels for endpoint blocks.
        endpoint_layers (Optional[Sequence[int]]): Endpoint stage indices.
        endpoint_scale_layer (int): Stage that defines the endpoint scale.
        use_norm (bool): Whether normalization layers are enabled.
        norm_layer (Optional[str]): Normalization layer factory name.
        norm_before (bool): Whether normalization precedes activation.
        upsampling_mode (str): Upsampling mode for endpoint alignment.
    """

    def __init__(
        self,
        in_channels: int = 1,
        in_conv_channels: int = 64,
        in_kernel_size: int = 3,
        in_stride: int = 1,
        resb_type: str = "basic",
        resb_repeats: List[int] = [2, 2, 2, 2],
        resb_channels: Union[int, Sequence[int]] = [64, 128, 256, 512],
        resb_kernel_sizes: Union[int, Sequence[int]] = 3,
        resb_strides: Union[int, Sequence[int]] = 2,
        resb_dilations: Union[int, Sequence[int]] = 1,
        resb_groups: int = 1,
        head_channels: int = 0,
        hid_act: Union[str, Dict[str, Any]] = "relu",
        head_act: Optional[Union[str, Dict[str, Any]]] = None,
        dropout_rate: float = 0,
        drop_connect_rate: float = 0,
        se_r: int = 16,
        time_se: bool = False,
        in_feats: Optional[int] = None,
        res2net_width_factor: float = 1,
        res2net_scale: int = 4,
        multilayer: bool = False,
        multilayer_concat: bool = False,
        endpoint_channels: Optional[int] = None,
        endpoint_layers: Optional[Sequence[int]] = None,
        endpoint_scale_layer: int = -1,
        use_norm: bool = True,
        norm_layer: Optional[str] = None,
        norm_before: bool = True,
        upsampling_mode: str = "nearest",
    ) -> None:
        """Build the encoder.

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
            drop_connect_rate: Drop-connect probability.
            se_r: Squeeze-excitation reduction ratio.
            time_se: Whether SE pooling is time-only.
            in_feats: Input feature dimension for `time_se`.
            res2net_width_factor: Res2Net width scaling factor.
            res2net_scale: Res2Net scale factor.
            multilayer: Enable multi-layer feature aggregation.
            multilayer_concat: Concatenate instead of averaging endpoints.
            endpoint_channels: Output channels for endpoint blocks.
            endpoint_layers: Endpoint stage indices.
            endpoint_scale_layer: Stage that defines the endpoint scale.
            use_norm: Whether to use normalization layers.
            norm_layer: Normalization layer factory name.
            norm_before: Whether normalization precedes activation.
            upsampling_mode: Upsampling mode for endpoint alignment.
        """

        super().__init__()

        self.resb_type = resb_type
        bargs = {}  # block's extra arguments
        if resb_type == "basic":
            self._block = ResNet2dBasicBlock
        elif resb_type == "bn":
            self._block = ResNet2dBNBlock
        elif resb_type == "sebasic":
            self._block = SEResNet2dBasicBlock
            bargs["se_r"] = se_r
        elif resb_type == "sebn":
            self._block = SEResNet2dBNBlock
            bargs["se_r"] = se_r
        elif resb_type in ["res2basic", "seres2basic", "res2bn", "seres2bn"]:
            bargs["width_factor"] = res2net_width_factor
            bargs["scale"] = res2net_scale
            if resb_type in ["seres2basic", "seres2bn"]:
                bargs["se_r"] = se_r
                bargs["time_se"] = time_se
            if resb_type in ["res2basic", "seres2basic"]:
                self._block = Res2Net2dBasicBlock
            else:
                self._block = Res2Net2dBNBlock
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
        self.drop_connect_rate = drop_connect_rate
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.se_r = se_r
        self.time_se = time_se
        self.in_feats = in_feats
        self.res2net_width_factor = res2net_width_factor
        self.res2net_scale = res2net_scale

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(min(self.resb_channels) // 2, 32)
            norm_groups = max(norm_groups, resb_groups)
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
        total_blocks = sum(self.resb_repeats)

        # middle blocks
        self.blocks = nn.ModuleList([])
        k = 0
        self.resb_scales = []
        for i in range(num_superblocks):
            blocks_i = nn.ModuleList([])
            repeats_i = self.resb_repeats[i]
            channels_i = self.resb_channels[i]
            stride_i = self.resb_strides[i]
            kernel_size_i = self.resb_kernel_sizes[i]
            dilation_i = self.resb_dilations[i]
            # if there is downsampling the dilation of the first block
            # is set to 1
            dilation_i1 = dilation_i if stride_i == 1 else 1
            if time_se:
                if self.in_feats is None:
                    raise ValueError("in_feats must be set when time_se=True")
                num_feats_i = int(self.in_feats / (self._downsample_factor * stride_i))
                bargs["num_feats"] = num_feats_i
            drop_i = (
                0.0
                if total_blocks <= 1
                else drop_connect_rate * k / (total_blocks - 1)
            )
            block_i = self._block(
                cur_in_channels,
                channels_i,
                kernel_size_i,
                stride=stride_i,
                dilation=dilation_i1,
                groups=self.resb_groups,
                activation=hid_act,
                dropout_rate=dropout_rate,
                drop_connect_rate=drop_i,
                use_norm=use_norm,
                norm_layer=self._norm_layer,
                norm_before=norm_before,
                **bargs
            )

            blocks_i.append(block_i)
            k += 1
            self._context += block_i.context * self._downsample_factor
            self._downsample_factor *= block_i.downsample_factor
            self.resb_scales.append(self._downsample_factor)

            for j in range(repeats_i - 1):
                drop_i = (
                    0.0
                    if total_blocks <= 1
                    else drop_connect_rate * k / (total_blocks - 1)
                )
                block_i = self._block(
                    channels_i,
                    channels_i,
                    kernel_size_i,
                    stride=1,
                    dilation=dilation_i,
                    groups=self.resb_groups,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    drop_connect_rate=drop_i,
                    use_norm=use_norm,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                    **bargs
                )

                blocks_i.append(block_i)
                k += 1
                self._context += block_i.context * self._downsample_factor

            self.blocks.append(blocks_i)
            cur_in_channels = channels_i

        if multilayer:
            if endpoint_layers is None:
                endpoint_layers = [i + 1 for i in range(num_superblocks)]

            if endpoint_channels is None:
                endpoint_channels = self.resb_channels[endpoint_scale_layer]

            self.is_endpoint = [
                True if i + 1 in endpoint_layers else False
                for i in range(num_superblocks)
            ]
            self.has_endpoint_block = [False] * num_superblocks
            self.endpoint_block_idx = [0] * num_superblocks
            endpoint_scale = self.resb_scales[endpoint_scale_layer]
            endpoint_blocks = nn.ModuleList([])
            cur_endpoint = 0
            in_concat_channels = 0
            for i in range(num_superblocks):
                if self.is_endpoint[i]:
                    if multilayer_concat:
                        out_channels = self.resb_channels[i]
                        if self.resb_scales[i] != endpoint_scale:
                            self.has_endpoint_block[i] = True

                        in_concat_channels += out_channels
                    else:
                        self.has_endpoint_block[i] = True
                        out_channels = endpoint_channels

                    if self.has_endpoint_block[i]:
                        endpoint_i = ResNet2dEndpoint(
                            self.resb_channels[i],
                            out_channels,
                            in_scale=self.resb_scales[i],
                            scale=endpoint_scale,
                            upsampling_mode=upsampling_mode,
                            activation=hid_act,
                            norm_layer=self._norm_layer,
                            norm_before=norm_before,
                        )
                        self.endpoint_block_idx[i] = cur_endpoint
                        endpoint_blocks.append(endpoint_i)
                        cur_endpoint += 1

            self.endpoint_blocks = endpoint_blocks
            if multilayer_concat:
                self.concat_endpoint_block = ResNet2dEndpoint(
                    in_concat_channels,
                    endpoint_channels,
                    in_scale=1,
                    scale=1,
                    activation=hid_act,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                )
        else:
            endpoint_channels = self.resb_channels[-1]

        self.multilayer = multilayer
        self.multilayer_concat = multilayer_concat
        self.endpoint_channels = endpoint_channels
        self.endpoint_layers = endpoint_layers
        self.endpoint_scale_layer = endpoint_scale_layer
        self.upsampling_mode = upsampling_mode

        # head feature block
        if self.head_channels > 0:
            head_in_channels = endpoint_channels if multilayer else cur_in_channels
            self.head_block = DC2dEncBlock(
                head_in_channels,
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
            Output spatial size after all encoder stages.
        """
        out_size = int((in_size - 1) // self.in_stride + 1)

        if self.multilayer:
            endpoint_idx = self.endpoint_scale_layer
            if endpoint_idx < 0:
                endpoint_idx += len(self.resb_strides)
            strides = self.resb_strides[: endpoint_idx + 1]
        else:
            strides = self.resb_strides

        for stride in strides:
            out_size = int((out_size - 1) // stride + 1)

        return out_size

    def in_context(self) -> Tuple[int, int]:
        """Return the receptive-field context required by the encoder.

        Returns:
            A symmetric `(left, right)` context tuple.
        """
        return (self._context, self._context)

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
            self.head_channels if self.head_channels > 0 else self.endpoint_channels
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

    @staticmethod
    def _update_mask(
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor],
        x_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build or reuse a time mask for a batch.

        Args:
            x: Input tensor.
            x_lengths: Sequence lengths for `x`.
            x_mask: Optional precomputed mask.

        Returns:
            A time mask or `None` when `x_lengths` is not provided.
        """
        if x_lengths is None:
            return None

        if x_mask is not None and x.size(-1) == x_mask.size(-1):
            return x_mask

        return seq_lengths_to_mask(x_lengths, x.size(-1), time_dim=3)

    @staticmethod
    def _match_lens(endpoints: List[torch.Tensor]) -> List[torch.Tensor]:
        """Center-crop endpoint tensors to shared spatial sizes.

        Args:
            endpoints: Endpoint tensors to align.

        Returns:
            Endpoint tensors cropped to the smallest height and width.
        """
        min_h = min(e.shape[-2] for e in endpoints)
        min_w = min(e.shape[-1] for e in endpoints)
        for i in range(len(endpoints)):
            if endpoints[i].shape[-2] > min_h:
                h_start = (endpoints[i].shape[-2] - min_h) // 2
                endpoints[i] = endpoints[i][:, :, h_start : h_start + min_h, :]

            if endpoints[i].shape[-1] > min_w:
                w_start = (endpoints[i].shape[-1] - min_w) // 2
                endpoints[i] = endpoints[i][:, :, :, w_start : w_start + min_w]

        return endpoints

    def forward(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the encoder forward pass.

        Args:
            x: Input tensor.
            x_lengths: Optional sequence lengths for masking.

        Returns:
            The encoded tensor.
        """

        x_mask = self._update_mask(x, x_lengths)
        x = self.in_block(x, x_mask=x_mask)
        endpoints = []
        for i, superblock in enumerate(self.blocks):
            for j, block in enumerate(superblock):
                if x_mask is not None and block.stride > 1:
                    x_mask = x_mask[..., :: block.stride]

                x = block(x, x_mask=x_mask)

            if self.multilayer and self.is_endpoint[i]:
                endpoint_i = x
                if self.has_endpoint_block[i]:
                    idx = self.endpoint_block_idx[i]
                    endpoint_i = self.endpoint_blocks[idx](endpoint_i)

                endpoints.append(endpoint_i)

        if self.multilayer:
            endpoints = self._match_lens(endpoints)
            if self.multilayer_concat:
                x = torch.cat(endpoints, dim=1)
                x = self.concat_endpoint_block(x)
            else:
                x = torch.mean(torch.stack(endpoints), 0)

        if self.head_channels > 0:
            x = self.head_block(x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return the serializable configuration.

        Args:
            no_class_name: If `True`, omit the class metadata from the base
                configuration.

        Returns:
            A dictionary with the encoder configuration.
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
            "drop_connect_rate": self.drop_connect_rate,
            "hid_act": hid_act,
            "head_act": head_act,
            "se_r": self.se_r,
            "time_se": self.time_se,
            "res2net_width_factor": self.res2net_width_factor,
            "res2net_scale": self.res2net_scale,
            "use_norm": self.use_norm,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
            "multilayer": self.multilayer,
            "multilayer_concat": self.multilayer_concat,
            "endpoint_channels": self.endpoint_channels,
            "endpoint_layers": self.endpoint_layers,
            "endpoint_scale_layer": self.endpoint_scale_layer,
            "upsampling_mode": self.upsampling_mode,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(
        self, override_dropouts: bool, dropout_rate: float, drop_connect_rate: float
    ) -> None:
        """Update dropout configuration from a pretrained model.

        Args:
            override_dropouts: Whether to apply the provided dropout values.
            dropout_rate: New dropout probability.
            drop_connect_rate: New drop-connect probability.
        """
        if override_dropouts:
            logging.info("chaning resnet2d dropouts")
            self.change_dropouts(dropout_rate, drop_connect_rate)

    def change_dropouts(self, dropout_rate: float, drop_connect_rate: float) -> None:
        """Update dropout and drop-connect probabilities.

        Args:
            dropout_rate: New dropout probability.
            drop_connect_rate: New drop-connect probability.
        """
        super().change_dropouts(dropout_rate)
        from ..layers import DropConnect2d

        for module in self.modules():
            if isinstance(module, DropConnect2d):
                if self.drop_connect_rate == 0:
                    module.p = drop_connect_rate
                else:
                    module.p *= drop_connect_rate / self.drop_connect_rate

        self.drop_connect_rate = drop_connect_rate
        self.dropout_rate = dropout_rate

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments accepted by the encoder constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            The subset accepted by :meth:`__init__`.
        """
        return filter_func_args(ResNet2dEncoder.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set()
    ) -> None:
        """Add command-line arguments for this encoder.

        Args:
            parser: Argument parser to extend.
            prefix: Optional argument namespace prefix.
            skip: Argument names to skip.
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
            "--resb-type",
            default="basic",
            choices=[
                "basic",
                "bn",
                "sebasic",
                "sebn",
                "res2basic",
                "res2bn",
                "seres2basic",
                "seres2bn",
            ],
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

        if "head_channels" not in skip:
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
                "--drop-connect-rate",
                default=0,
                type=float,
                help="layer drop probability",
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

        parser.add_argument(
            "--time-se",
            default=False,
            action="store_true",
            help=("squeeze-excitation pooling is done in time dimension only"),
        )

        parser.add_argument(
            "--res2net-width-factor",
            default=1,
            type=float,
            help=(
                "scaling factor for channels in middle layer "
                "of res2net bottleneck blocks"
            ),
        )

        parser.add_argument(
            "--res2net-scale",
            default=1,
            type=int,
            help=("res2net scaling parameter "),
        )

        parser.add_argument(
            "--multilayer",
            default=False,
            action="store_true",
            help="use multilayer feature aggregation (mfa)",
        )

        parser.add_argument(
            "--multilayer-concat",
            default=False,
            action="store_true",
            help="use concatenation for mfa",
        )

        parser.add_argument(
            "--endpoint-channels",
            default=None,
            type=int,
            help=("num. endpoint channels when using mfa"),
        )

        parser.add_argument(
            "--endpoint-layers",
            default=None,
            nargs="+",
            type=int,
            help=(
                "layers to aggreagate in mfa, "
                "if None, all residual blocks are aggregated"
            ),
        )

        parser.add_argument(
            "--endpoint-scale-layer",
            default=-1,
            type=int,
            help=("layer number which indicates the time scale in mfa"),
        )

        parser.add_argument(
            "--upsampling-mode",
            choices=["nearest", "bilinear"],
            default="nearest",
            help=("upsampling method when upsampling feature maps for mfa"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
