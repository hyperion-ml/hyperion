"""
Copyright 2020 Magdalena Rybicka
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Conv1d, Linear

from ..layer_blocks import (
    BlockSpec,
    Res2NetBasicBlock,
    Res2NetBNBlock,
    ResNetBasicBlock,
    ResNetBNBlock,
    ResNetInputBlock,
    SpineConv,
    SpineEndpoints,
    SpineResample,
)
from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from .net_arch import NetArch

SPINENET_BLOCK_SPECS = [
    # level, block type, tuple of inputs, is output
    (2, ResNetBNBlock, (None, None), False),
    (2, ResNetBNBlock, (None, None), False),
    (2, ResNetBNBlock, (0, 1), False),
    (4, ResNetBasicBlock, (0, 1), False),
    (3, ResNetBNBlock, (2, 3), False),
    (4, ResNetBNBlock, (2, 4), False),
    (6, ResNetBasicBlock, (3, 5), False),
    (4, ResNetBNBlock, (3, 5), False),
    (5, ResNetBasicBlock, (6, 7), False),
    (7, ResNetBasicBlock, (6, 8), False),
    (5, ResNetBNBlock, (8, 9), False),
    (5, ResNetBNBlock, (8, 10), False),
    (4, ResNetBNBlock, (5, 10), True),
    (3, ResNetBNBlock, (4, 10), True),
    (5, ResNetBNBlock, (7, 12), True),
    (7, ResNetBNBlock, (5, 14), True),
    (6, ResNetBNBlock, (12, 14), True),
]

R0_SP53_BLOCK_SPECS = [
    # level, block type, tuple of inputs, is output
    (2, ResNetBNBlock, (None, None), False),  # 0
    (2, ResNetBNBlock, (None, None), False),  # 1
    (2, ResNetBNBlock, (0, 1), False),  # 2
    (3, ResNetBNBlock, (0, 1), False),  # 3
    (3, ResNetBNBlock, (2, 3), False),  # 4
    (4, ResNetBNBlock, (2, 4), False),  # 5
    (4, ResNetBNBlock, (3, 5), False),  # 6
    (3, ResNetBNBlock, (5, 6), False),  # 7
    (5, ResNetBNBlock, (4, 7), False),  # 8
    (4, ResNetBNBlock, (4, 8), False),  # 9
    (4, ResNetBNBlock, (8, 9), False),  # 10
    (4, ResNetBNBlock, (8, 10), False),  # 11
    (3, ResNetBNBlock, (4, 10), True),  # 12
    (4, ResNetBNBlock, (6, 7), True),  # 13 it has 3 inputs
    (5, ResNetBNBlock, (8, 13), True),  # 14
    (7, ResNetBNBlock, (6, 9), True),  # 15
    (6, ResNetBNBlock, (7, 9), True),  # 16
]

SPINENET_BLOCK_SPECS_5 = [
    # level, block type, tuple of inputs, is output
    (2, ResNetBNBlock, (None, None), False),  # 0
    (2, ResNetBNBlock, (None, None), False),  # 1
    (2, ResNetBNBlock, (0, 1), False),  # 2
    (4, ResNetBasicBlock, (0, 1), False),  # 3
    (3, ResNetBNBlock, (2, 3), False),  # 4
    (4, ResNetBNBlock, (2, 4), False),  # 5
    (6, ResNetBasicBlock, (3, 5), False),  # 6
    (4, ResNetBNBlock, (3, 5), False),  # 7
    (5, ResNetBasicBlock, (6, 7), False),  # 8
    (7, ResNetBasicBlock, (6, 8), False),  # 9
    (5, ResNetBNBlock, (8, 9), False),  # 10
    (5, ResNetBNBlock, (8, 10), False),  # 11
    (4, ResNetBNBlock, (5, 10), True),  # 12
    # (3, ResNetBNBlock, (4, 10), True),      # 13
    (5, ResNetBNBlock, (7, 12), True),  # 14
    # (7, ResNetBNBlock, (5, 14), True),      # 15
    # (6, ResNetBNBlock, (12, 14), True),     # 16
]

FILTER_SIZE_MAP = {
    # level: channel multiplier
    1: 0.5,
    2: 1,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
}


class SpineNet(NetArch):
    """SpineNet backbone for 2D feature tensors.

    Attributes:
        in_channels: Number of input channels.
        output_levels: Output pyramid levels returned by the network.
        endpoints_num_filters: Channel width used by endpoint projections.
        feature_output_level: Spatial level used to align endpoint features.
        with_output: Whether the network includes a final classification head.
    """

    def __init__(
        self,
        in_channels: int,
        block_specs: Optional[Sequence[Any]] = None,
        output_levels: Sequence[int] = [3, 4, 5, 6, 7],
        endpoints_num_filters: int = 256,
        resample_alpha: float = 0.5,
        feature_output_level: Optional[int] = None,
        block_repeats: int = 1,
        filter_size_scale: float = 1.0,
        conv_channels: int = 64,
        base_channels: int = 64,
        out_units: int = 0,
        concat: bool = False,
        do_endpoint_conv: bool = True,
        concat_ax: int = 3,
        upsampling_type: str = "nearest",
        hid_act: Any = {"name": "relu", "inplace": True},
        out_act: Any = None,
        in_kernel_size: int = 7,
        in_stride: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        dropout_rate: float = 0,
        norm_layer: Optional[str] = None,
        norm_before: bool = True,
        do_maxpool: bool = True,
        in_norm: bool = True,
        in_feats: Optional[int] = None,
        se_r: int = 16,
        time_se: bool = False,
        has_se: bool = False,
        is_res2net: bool = False,
        res2net_scale: int = 4,
        res2net_width_factor: int = 1,
    ) -> None:
        """
        Base class for SpineNet structures.

        This implementation follows the paper "SpineNet: Learning
        Scale-Permuted Backbone for Recognition and Localization".

        Args:
            in_channels: Number of input channels.
            block_specs: Building-block specification. Each entry defines the
                block level, block type, input offsets, and output flag.
            output_levels: Levels whose outputs are exposed by the backbone.
            endpoints_num_filters: Channel width used in the endpoint blocks.
            resample_alpha: Resampling interpolation factor.
            feature_output_level: Level used to align endpoint feature sizes.
            block_repeats: Number of times each block is repeated.
            filter_size_scale: Multiplier applied to the base channel counts.
            conv_channels: Number of channels in the stem convolution.
            base_channels: Base width for the permuted blocks.
            out_units: Output head size; ``0`` disables the head.
            concat: If ``True``, concatenate endpoint tensors; otherwise mean.
            do_endpoint_conv: If ``True``, project endpoints to a common width.
            concat_ax: Concatenation axis when ``concat`` is enabled.
            upsampling_type: Upsampling mode for cross-scale resampling.
            hid_act: Hidden activation specification.
            out_act: Optional output activation specification.
            in_kernel_size: Kernel size of the first convolution.
            in_stride: Stride of the first convolution.
            zero_init_residual: If ``True``, zero-initialize residual branches.
            groups: Number of grouped-convolution groups in residual blocks.
            dropout_rate: Dropout probability used in residual blocks.
            norm_layer: Normalization layer name or alias.
            norm_before: If ``True``, apply normalization before activation.
            do_maxpool: If ``True``, keep the stem max-pooling layer.
            in_norm: If ``True``, normalize the input tensor before the stem.
            in_feats: Input feature size used by time-SE variants.
            se_r: Squeeze-excitation reduction ratio.
            time_se: If ``True``, use time-aware squeeze-excitation.
            has_se: If ``True``, enable squeeze-excitation blocks.
            is_res2net: If ``True``, build Res2Net-style residual blocks.
            res2net_scale: Res2Net scale factor.
            res2net_width_factor: Res2Net internal width multiplier.
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.base_channels = base_channels
        self.out_units = out_units
        self.endpoints_num_filters = endpoints_num_filters
        self.resample_alpha = resample_alpha
        self.block_repeats = block_repeats
        self.filter_size_scale = filter_size_scale
        self.concat = concat
        self.concat_ax = concat_ax
        self.do_endpoint_conv = do_endpoint_conv
        self.feature_output_level = (
            min(output_levels) if feature_output_level is None else feature_output_level
        )

        self.res2net_scale = res2net_scale
        self.res2net_width_factor = res2net_width_factor
        self.is_res2net = is_res2net

        self.se_r = se_r
        self.time_se = time_se
        self.has_se = has_se

        self._block_specs = (
            BlockSpec.build_block_specs(SPINENET_BLOCK_SPECS)
            if block_specs is None
            else BlockSpec.build_block_specs(block_specs)
        )
        self.output_levels = output_levels
        self.upsampling_type = upsampling_type
        self.dilation = 1

        self.hid_act = hid_act
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        self.groups = groups
        self.norm_before = norm_before
        self.do_maxpool = do_maxpool
        self.dropout_rate = dropout_rate
        self.in_norm = in_norm
        self.in_feats = in_feats

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(base_channels // 2, 32)
            norm_groups = max(norm_groups, groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        self.in_bn: Optional[nn.Module] = None
        if in_norm:
            self.in_bn = self._norm_layer(in_channels)

        self.in_block = ResNetInputBlock(
            in_channels,
            conv_channels,
            kernel_size=in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            norm_layer=self._norm_layer,
            norm_before=norm_before,
            do_maxpool=do_maxpool,
        )

        if self.is_res2net:
            if self._block_specs[0].block_fn == ResNetBNBlock:
                _in_block = Res2NetBNBlock
            elif self._block_specs[0].block_fn == ResNetBasicBlock:
                _in_block = Res2NetBasicBlock
        else:
            _in_block = self._block_specs[0].block_fn

        self.stem0 = self._make_layer(
            _in_block, 2, self.block_repeats, in_channels=conv_channels
        )
        self.stem1 = self._make_layer(_in_block, 2, self.block_repeats)

        self.stem_nbr = 2  # the number of the stem layers
        self.blocks = self._make_permuted_blocks(self._block_specs[self.stem_nbr :])
        self.connections = self._make_permuted_connections(
            self._block_specs[self.stem_nbr :]
        )
        self.endpoints = self._make_endpoints()

        self._context = self._compute_max_context(self.in_block.context)
        self._downsample_factor = self.in_block.downsample_factor * 2 ** (
            self.feature_output_level - 2
        )
        self.with_output = False
        self.out_act = None
        if out_units > 0:
            self.with_output = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            cur_channels = self._compute_channel_size()
            self.output = nn.Linear(cur_channels, out_units)
            self.out_act = AF.create(out_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = "relu"
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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        self.zero_init_residual = zero_init_residual
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBNBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_permuted_blocks(self, block_specs: Sequence[Any]) -> nn.ModuleList:
        """
        Build the residual block stack for the permuted backbone.

        Args:
            block_specs: Block specifications after the stem stages.

        Returns:
            nn.ModuleList: Sequential block modules in execution order.
        """
        blocks = nn.ModuleList([])
        for block in block_specs:
            if self.is_res2net:
                if block.block_fn == ResNetBNBlock:
                    _block = Res2NetBNBlock
                elif block.block_fn == ResNetBasicBlock:
                    _block = Res2NetBasicBlock
                else:
                    _block = block.block_fn
            else:
                _block = block.block_fn
            layer_i = self._make_layer(_block, block.level, self.block_repeats)
            blocks.append(layer_i)
        return blocks

    def _make_permuted_connections(self, block_specs: Sequence[Any]) -> nn.ModuleList:
        """
        Build the cross-scale resampling paths between blocks.

        Args:
            block_specs: Block specifications after the stem stages.

        Returns:
            nn.ModuleList: Connection modules aligned with the block list.
        """
        connections = nn.ModuleList([])
        for block in block_specs:
            expansion = block.block_fn.expansion
            out_channels = (
                int(
                    FILTER_SIZE_MAP[block.level]
                    * self.filter_size_scale
                    * self.base_channels
                )
                * expansion
            )

            connections_i = nn.ModuleList([])
            for i in block.input_offsets:
                offset_block = self._block_specs[i]
                scale = offset_block.level - block.level
                in_channels = int(
                    FILTER_SIZE_MAP[offset_block.level]
                    * self.filter_size_scale
                    * self.base_channels
                )
                connections_i.append(
                    SpineResample(
                        offset_block,
                        in_channels,
                        out_channels,
                        scale,
                        self.resample_alpha,
                        self.upsampling_type,
                        activation=self.hid_act,
                        norm_layer=self._norm_layer,
                        norm_before=self.norm_before,
                    )
                )
            connections_i.append(AF.create(self.hid_act))
            connections.append(connections_i)
        return connections

    def _make_endpoints(self) -> nn.ModuleDict:
        """
        Build the endpoint projection modules.

        Returns:
            nn.ModuleDict: Endpoint blocks keyed by output level.
        """
        endpoints = nn.ModuleDict()
        for block_spec in self._block_specs:
            if block_spec.is_output and block_spec.level in self.output_levels:
                expansion = block_spec.block_fn.expansion
                in_channels = (
                    int(
                        FILTER_SIZE_MAP[block_spec.level]
                        * self.filter_size_scale
                        * self.base_channels
                    )
                    * expansion
                )
                out_channels = (
                    self.endpoints_num_filters if self.do_endpoint_conv else in_channels
                )
                endpoints[str(block_spec.level)] = SpineEndpoints(
                    in_channels,
                    out_channels,
                    block_spec.level,
                    self.feature_output_level,
                    self.upsampling_type,
                    activation=self.hid_act,
                    norm_layer=self._norm_layer,
                    norm_before=self.norm_before,
                    do_endpoint_conv=self.do_endpoint_conv,
                )

        return endpoints

    def _make_layer(
        self,
        block: type,
        block_level: int,
        num_blocks: int,
        in_channels: Optional[int] = None,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Build a residual layer for one SpineNet level.

        Args:
            block: Residual block class to instantiate.
            block_level: Target pyramid level for the layer.
            num_blocks: Number of repeated blocks to stack.
            in_channels: Optional input channel override for the first block.
            stride: Spatial stride used by the first block when not dilated.
            dilate: If ``True``, replace the stride with dilation.

        Returns:
            nn.Sequential: The constructed residual layer.
        """

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        kwargs = {}
        if self.has_se:
            if self.time_se:
                num_feats = int(self.in_feats / self.in_block.downsample_factor)
                for i in range(block_level - 2):
                    num_feats = (
                        int(num_feats // 2)
                        if num_feats % 2 == 0
                        else int(num_feats // 2 + 1)
                    )
                kwargs = {"se_r": self.se_r, "time_se": True, "num_feats": num_feats}
            else:
                kwargs = {"se_r": self.se_r}

        if self.is_res2net and block != ResNetBasicBlock:
            kwargs["scale"] = self.res2net_scale
            kwargs["width_factor"] = self.res2net_width_factor
        channels = int(
            FILTER_SIZE_MAP[block_level] * self.base_channels * self.filter_size_scale
        )
        if in_channels is None:
            in_channels = channels * block.expansion

        layers = []
        layers.append(
            block(
                in_channels,
                channels,
                activation=self.hid_act,
                stride=stride,
                dropout_rate=self.dropout_rate,
                groups=self.groups,
                dilation=previous_dilation,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
                **kwargs,
            )
        )

        cur_in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    cur_in_channels,
                    channels,
                    activation=self.hid_act,
                    dropout_rate=self.dropout_rate,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                    norm_before=self.norm_before,
                    **kwargs,
                )
            )

        return nn.Sequential(*layers)

    def _compute_max_context(self, in_context: int) -> int:
        """
        Compute the maximum context consumed by the network.

        Args:
            in_context: Context contributed by the input stem block.

        Returns:
            int: Maximum receptive-field context.
        """
        block_context = {  # we can define specific values as inside the network the dilation or stride is not applied
            ResNetBNBlock: 1,
            ResNetBasicBlock: 2,
        }
        base_downsample_factor = self.in_block.downsample_factor
        context0 = in_context
        # context of the first two blocks (stem part)
        context0 += (
            base_downsample_factor
            * block_context[self._block_specs[0].block_fn]
            * self.block_repeats
        )
        context1 = (
            context0
            + base_downsample_factor
            * block_context[self._block_specs[1].block_fn]
            * self.block_repeats
        )
        contexts = [context0, context1]

        # context in the scale permuted part
        num_outgoing_connections = [0, 0]
        for idx, block in enumerate(self._block_specs[self.stem_nbr :]):
            input0 = block.input_offsets[0]
            input1 = block.input_offsets[1]

            target_level = block.level
            # we add context if in the resampling connection was downsampling operation (it includes 3x3 convolution)
            resample0 = (
                self._block_specs[input0].level + 1
                if self._block_specs[input0].level - target_level < 0
                else 0
            )
            resample1 = (
                self._block_specs[input1].level + 1
                if self._block_specs[input1].level - target_level < 0
                else 0
            )
            parent0_context = contexts[input0] + resample0
            parent1_context = contexts[input1] + resample1
            # as input context we choose the input with higher value
            target_context = max(parent0_context, parent1_context)

            num_outgoing_connections[input0] += 1
            num_outgoing_connections[input1] += 1
            # Connect intermediate blocks with outdegree 0 to the output block.
            # Some blocks have also this additional connection
            if block.is_output:
                for j, j_connections in enumerate(num_outgoing_connections):
                    if (
                        j_connections == 0
                        and self._block_specs[j].level == target_level
                    ):
                        target_context = max(contexts[j], target_context)
                        num_outgoing_connections[j] += 1

            downsample_factor = base_downsample_factor * 2 ** (target_level - 2)
            target_context += (
                block_context[block.block_fn] * self.block_repeats * downsample_factor
            )
            contexts.append(target_context)
            num_outgoing_connections.append(0)
        # logging.info('block\'s contexts: {}'.format(contexts))
        return max(contexts)

    def _compute_out_size(self, in_size: int) -> int:
        """Compute the output spatial size for one axis.

        Args:
            in_size: Input size of the height or width dimension.

        Returns:
            int: Output size after all downsampling stages.
        """
        out_size = int((in_size - 1) // self.in_stride + 1)
        if self.do_maxpool:
            out_size = int((out_size - 1) // 2 + 1)

        downsample_levels = self.feature_output_level - 2
        for i in range(downsample_levels):
            out_size = (
                int(out_size // 2) if out_size % 2 == 0 else int(out_size // 2 + 1)
            )

        return out_size

    def _compute_channel_size(self) -> int:
        """
        Compute the number of channels produced by the endpoint stack.

        Returns:
            int: Output channel count.
        """

        if not self.do_endpoint_conv:
            C = 0
            for output_level in self.output_levels:
                C += self.base_channels * 4 * FILTER_SIZE_MAP[output_level]
            return C
        else:
            if self.concat and self.concat_ax == 1:
                C = len(self.output_levels) * self.endpoints_num_filters
                return C
        return self.endpoints_num_filters

    # def in_context(self):
    #     """
    #     Returns:
    #       Tuple (past, future) context required to predict one frame.
    #     """
    #     return (self._context, self._context)

    def in_shape(self) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
        """
        Return the expected input shape.

        Returns:
            Tuple[Optional[int], int, Optional[int], Optional[int]]: Input
            tensor shape specification.
        """
        return (None, self.in_channels, None, None)

    def out_shape(
        self, in_shape: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Any, ...]:
        """Compute the output shape given an input shape.

        Args:
            in_shape: Optional input shape. When omitted, only the channel
                dimension is reported.

        Returns:
            Tuple[Any, ...]: Output tensor shape specification. This is a
            2-tuple when a classification head is present and a 4-tuple
            otherwise.
        """

        if self.with_output:
            return (None, self.out_units)

        if in_shape is None:
            return (None, self.endpoints_num_filters, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = None
        else:
            H = self._compute_out_size(in_shape[2])

            # in case of concatenation along feature dimension
            if self.concat_ax == 2 and self.concat:
                H = H * len(self.output_levels)

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])

        C = self._compute_channel_size()

        return (in_shape[0], C, H, W)

    def _match_shape(
        self, x: torch.Tensor, target_shape: Sequence[int]
    ) -> torch.Tensor:
        """Crop a tensor so its spatial dimensions match a target shape.

        Args:
            x: Input tensor to crop.
            target_shape: Desired trailing dimensions.

        Returns:
            torch.Tensor: Cropped tensor with contiguous memory layout.
        """
        x_dim = x.dim()
        ddim = x_dim - len(target_shape)
        for i in range(2, x_dim):
            surplus = x.size(i) - target_shape[i - ddim]

            assert surplus >= 0
            if surplus > 0:
                x = torch.narrow(x, i, surplus // 2, target_shape[i - ddim])

        return x.contiguous()

    def _match_feat_shape(
        self, feat0: torch.Tensor, feat1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match the spatial shape of two feature maps.

        Args:
            feat0: First feature tensor.
            feat1: Second feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature tensors with matched
            spatial sizes.
        """
        surplus = feat1.size(3) - feat0.size(3)
        if surplus >= 0:
            feat1 = self._match_shape(feat1, list(feat0.size())[2:])
        else:
            feat0 = self._match_shape(feat0, list(feat1.size())[2:])
        return feat0, feat1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, channels, height, width)``.

        Returns:
            torch.Tensor: Output logits when ``out_units > 0``; otherwise the
            final feature tensor.
        """

        if self.in_norm and self.in_bn is not None:
            x = self.in_bn(x)

        x = self.in_block(x)

        feat0 = self.stem0(x)
        feat1 = self.stem1(feat0)
        feats = [feat0, feat1]

        output_feats = {}
        num_outgoing_connections = [0, 0]
        for idx, block in enumerate(self._block_specs[self.stem_nbr :]):
            input0 = block.input_offsets[0]
            input1 = block.input_offsets[1]

            parent0_feat = self.connections[idx][0](feats[input0])
            parent1_feat = self.connections[idx][1](feats[input1])
            parent0_feat, parent1_feat = self._match_feat_shape(
                parent0_feat, parent1_feat
            )
            target_feat = parent0_feat + parent1_feat

            num_outgoing_connections[input0] += 1
            num_outgoing_connections[input1] += 1
            # Connect intermediate blocks with outdegree 0 to the output block.
            if block.is_output:
                for j, (j_feat, j_connections) in enumerate(
                    zip(feats, num_outgoing_connections)
                ):
                    if j_connections == 0 and j_feat.shape == target_feat.shape:
                        target_feat += j_feat
                        num_outgoing_connections[j] += 1

            target_feat = self.connections[idx][2](
                target_feat
            )  # pass input through the activation function
            x = self.blocks[idx](target_feat)

            feats.append(x)
            num_outgoing_connections.append(0)
            if block.is_output and block.level in self.output_levels:
                if str(block.level) in output_feats:
                    raise ValueError(
                        "Duplicate feats found for output level {}.".format(block.level)
                    )
                output_feats[str(block.level)] = x

        output_endpoints = []
        output_shape = list(
            output_feats[str(self.feature_output_level)].size()
        )  # get the target output size

        for endpoint in self.endpoints:
            if self.endpoints[endpoint] is not None:
                endpoint_i = self.endpoints[endpoint](output_feats[endpoint])
            else:
                endpoint_i = output_feats[endpoint]
            endpoint_i = self._match_shape(endpoint_i, output_shape)
            output_endpoints.append(endpoint_i)

        if self.concat:
            x = torch.cat(output_endpoints, self.concat_ax)
        else:
            x = torch.mean(torch.stack(output_endpoints), 0)

        if self.with_output:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)
        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Get a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the class name from the base
                configuration returned by :class:`NetArch`.

        Returns:
            Dict[str, Any]: Configuration dictionary for reconstruction.
        """

        out_act = AF.get_config(self.out_act)
        hid_act = self.hid_act

        config = {
            "in_channels": self.in_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "conv_channels": self.conv_channels,
            "base_channels": self.base_channels,
            "endpoints_num_filters": self.endpoints_num_filters,
            "resample_alpha": self.resample_alpha,
            "block_repeats": self.block_repeats,
            "filter_size_scale": self.filter_size_scale,
            "output_levels": self.output_levels,
            "feature_output_level": self.feature_output_level,
            "out_units": self.out_units,
            "concat": self.concat,
            "concat_ax": self.concat_ax,
            "do_endpoint_conv": self.do_endpoint_conv,
            "upsampling_type": self.upsampling_type,
            "zero_init_residual": self.zero_init_residual,
            "groups": self.groups,
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
            "in_norm": self.in_norm,
            "do_maxpool": self.do_maxpool,
            "out_act": out_act,
            "hid_act": hid_act,
            "se_r": self.se_r,
            "in_feats": self.in_feats,
            "res2net_scale": self.res2net_scale,
            "res2net_width_factor": self.res2net_width_factor,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))


# SpineNet structures from the original paper
class SpineNet49(SpineNet):
    """SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 256
        kwargs["filter_size_scale"] = 1.0
        kwargs["resample_alpha"] = 0.5
        kwargs["block_repeats"] = 1
        super(SpineNet49, self).__init__(in_channels, **kwargs)


class SpineNet49S(SpineNet):
    """Smaller SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the smaller SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["resample_alpha"] = 0.5
        kwargs["block_repeats"] = 1
        super(SpineNet49S, self).__init__(in_channels, **kwargs)


class SpineNet96(SpineNet):
    """SpineNet-96 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the SpineNet-96 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 256
        kwargs["filter_size_scale"] = 1.0
        kwargs["resample_alpha"] = 0.5
        kwargs["block_repeats"] = 2
        super(SpineNet96, self).__init__(in_channels, **kwargs)


class SpineNet143(SpineNet):
    """SpineNet-143 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the SpineNet-143 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 256
        kwargs["filter_size_scale"] = 1.0
        kwargs["resample_alpha"] = 1.0
        kwargs["block_repeats"] = 3
        super(SpineNet143, self).__init__(in_channels, **kwargs)


class SpineNet190(SpineNet):
    """SpineNet-190 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the SpineNet-190 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 512
        kwargs["filter_size_scale"] = 1.3
        kwargs["resample_alpha"] = 1.0
        kwargs["block_repeats"] = 4
        super(SpineNet190, self).__init__(in_channels, **kwargs)


# SpineNet modifications
# Light SpineNets
class LSpineNet49(SpineNet):
    """Light-weight SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the light-weight SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super(LSpineNet49, self).__init__(in_channels, **kwargs)


class LSpineNet49_subpixel(SpineNet):
    """Light-weight SpineNet-49 variant with subpixel upsampling.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the subpixel-upsampled SpineNet-49 variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["upsampling_type"] = "subpixel"
        super(LSpineNet49_subpixel, self).__init__(in_channels, **kwargs)


class LSpineNet49_bilinear(SpineNet):
    """Light-weight SpineNet-49 variant with bilinear upsampling.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the bilinear-upsampled SpineNet-49 variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["upsampling_type"] = "bilinear"
        super(LSpineNet49_bilinear, self).__init__(in_channels, **kwargs)


class LSpineNet49_5(SpineNet):
    """Light-weight SpineNet variant exposing only level 5.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the single-output SpineNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["output_levels"] = [5]
        kwargs["do_endpoint_conv"] = False
        kwargs["block_specs"] = SPINENET_BLOCK_SPECS_5
        super(LSpineNet49_5, self).__init__(in_channels, **kwargs)


class LSpine2Net49(SpineNet):
    """Light-weight Res2Net-style SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the light-weight Res2Net SpineNet-49 variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["is_res2net"] = True
        super(LSpine2Net49, self).__init__(in_channels, **kwargs)


# Spine2Nets ans(Time-)Squeeze-and-Excitation
class SELSpine2Net49(SpineNet):
    """Light-weight Res2Net + SE SpineNet-49 variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the light-weight Res2Net + SE variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        super(SELSpine2Net49, self).__init__(in_channels, **kwargs)


class TSELSpine2Net49(SpineNet):
    """Light-weight Res2Net + time-SE SpineNet-49 variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the light-weight Res2Net + time-SE variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        kwargs["time_se"] = True
        super(TSELSpine2Net49, self).__init__(in_channels, **kwargs)


class Spine2Net49(SpineNet):
    """Res2Net-style SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the Res2Net SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["is_res2net"] = True
        super(Spine2Net49, self).__init__(in_channels, **kwargs)


class SESpine2Net49(SpineNet):
    """Res2Net + SE SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the Res2Net + SE SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        super(SESpine2Net49, self).__init__(in_channels, **kwargs)


class TSESpine2Net49(SpineNet):
    """Res2Net + time-SE SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the Res2Net + time-SE SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        kwargs["time_se"] = True
        super(TSESpine2Net49, self).__init__(in_channels, **kwargs)


class Spine2Net49S(SpineNet):
    """Smaller Res2Net-style SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the smaller Res2Net SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["is_res2net"] = True
        super(Spine2Net49S, self).__init__(in_channels, **kwargs)


class SESpine2Net49S(SpineNet):
    """Smaller Res2Net + SE SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the smaller Res2Net + SE SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        super(SESpine2Net49S, self).__init__(in_channels, **kwargs)


class TSESpine2Net49S(SpineNet):
    """Smaller Res2Net + time-SE SpineNet-49 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the smaller Res2Net + time-SE SpineNet-49 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        kwargs["time_se"] = True
        super(TSESpine2Net49S, self).__init__(in_channels, **kwargs)


# R0-SP53 (structure from the paper)
class LR0_SP53(SpineNet):
    """Light-weight R0-SP53 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the light-weight R0-SP53 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["block_specs"] = R0_SP53_BLOCK_SPECS
        super(LR0_SP53, self).__init__(in_channels, **kwargs)


class R0_SP53(SpineNet):
    """R0-SP53 configuration variant.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the R0-SP53 configuration.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["block_specs"] = R0_SP53_BLOCK_SPECS
        super(R0_SP53, self).__init__(in_channels, **kwargs)


# concatenation
class SpineNet49_concat_time(SpineNet):
    """SpineNet-49 variant that concatenates endpoint outputs.

    Attributes:
        None: This class only overrides the base SpineNet configuration.
    """

    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize the concatenating SpineNet-49 variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional SpineNet keyword arguments.
        """
        kwargs["concat"] = True
        super(SpineNet49_concat_time, self).__init__(in_channels, **kwargs)
