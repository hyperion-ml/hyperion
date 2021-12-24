"""
 Copyright 2020 Magdalena Rybicka
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from ..layer_blocks import ResNetInputBlock, ResNetBasicBlock, ResNetBNBlock
from ..layer_blocks import Res2NetBNBlock, Res2NetBasicBlock
from ..layer_blocks import BlockSpec, SpineResample, SpineEndpoints, SpineConv
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
    def __init__(
        self,
        in_channels,
        block_specs=None,
        output_levels=[3, 4, 5, 6, 7],
        endpoints_num_filters=256,
        resample_alpha=0.5,
        feature_output_level=None,
        block_repeats=1,
        filter_size_scale=1.0,
        conv_channels=64,
        base_channels=64,
        out_units=0,
        concat=False,
        do_endpoint_conv=True,
        concat_ax=3,
        upsampling_type="nearest",
        hid_act={"name": "relu6", "inplace": True},
        out_act=None,
        in_kernel_size=7,
        in_stride=2,
        zero_init_residual=False,
        groups=1,
        dropout_rate=0,
        norm_layer=None,
        norm_before=True,
        do_maxpool=True,
        in_norm=True,
        in_feats=None,
        se_r=16,
        time_se=False,
        has_se=False,
        is_res2net=False,
        res2net_scale=4,
        res2net_width_factor=1,
    ):
        """
        Base class for the SpineNet structure. Based on the paper
        SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization
        Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V. Le, Xiaodan Song
        https://arxiv.org/abs/1912.05027

        :param in_channels: nbr of channels of the input
        :param block_specs: specification of the building blocks: their type, input connections and information if block
        is an output
        :param output_levels: the output levels of the blocks that are taken as an output of the SpineNet
        :param endpoints_num_filters: the base number of channels out the SpineNet output
        :param resample_alpha: parameter for resampling connections
        :param concat: bool that decides wheter the outputs are concatenated or averaged
        :param do_endpoint_conv: bool that decides whether to do the projection of the output blocks to the common
        number of channels (the value of the number is the endpoints_num_filters)
        :param concat_ax: the axis along which perform the concatenation (if the concatenation is chosen)
        :param feature_output_level: the level that the output feature map sizes are resampled to (by default the target
        size is the biggest feature map)
        :param filter_size_scale: SpineNet parameter, that additionally rescales the number of channels of the SpineNet
        blocks(needed for bigger structures like SpineNet96 and higher or for SpineNet49S)
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

        if in_norm:
            self.in_bn = norm_layer(in_channels)

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

    def _make_permuted_blocks(self, block_specs):
        """
        Builds the blocks of the SpineNet structure.
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

    def _make_permuted_connections(self, block_specs):
        """
        Builds the cross-scale connections between the blocks.
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

    def _make_endpoints(self):
        """
        Builds the output endpoint blocks. In this part, the block outputs are forwarded through the 1x1 convs
        to the common number of channels (endpoints_num_filters) and feature maps are resized to the size of the
        feature_output_level.
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
        self, block, block_level, num_blocks, in_channels=None, stride=1, dilate=False
    ):

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
                **kwargs
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
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _compute_max_context(self, in_context):
        """
        Computes maximum possible context in the structure. The method may need a deeper revision.
        :param in_context: context from the input residual block.
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

    def _compute_out_size(self, in_size):
        """Computes output size given input size.
           Output size is not the same as input size because of
           downsampling steps.

        Args:
           in_size: input size of the H or W dimensions

        Returns:
           output_size
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

    def _compute_channel_size(self):
        """
        Returns:
          If the 1x1 conv is not conducted in the endpoint blocks, the number of channels is equal to the sum of the
          nbr of channels of the output blocks.
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

    def in_shape(self):
        """
        Returns:
          Tuple describing input shape for the network
        """
        return (None, self.in_channels, None, None)

    def out_shape(self, in_shape=None):
        """Computes the output shape given the input shape
        #
        #     Args:
        #       in_shape: input shape
        #     Returns:
        #       Tuple describing output shape for the network
        #"""

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

    def _match_shape(self, x, target_shape):
        x_dim = x.dim()
        ddim = x_dim - len(target_shape)
        for i in range(2, x_dim):
            surplus = x.size(i) - target_shape[i - ddim]

            assert surplus >= 0
            if surplus > 0:
                x = torch.narrow(x, i, surplus // 2, target_shape[i - ddim])

        return x.contiguous()

    def _match_feat_shape(self, feat0, feat1):
        """
        Match shape between feats of the input connections.
        """
        surplus = feat1.size(3) - feat0.size(3)
        if surplus >= 0:
            feat1 = self._match_shape(feat1, list(feat0.size())[2:])
        else:
            feat0 = self._match_shape(feat0, list(feat1.size())[2:])
        return feat0, feat1

    def forward(self, x, use_amp=False):
        if use_amp:
            with torch.cuda.amp.autocast():
                return self._forward(x)

        return self._forward(x)

    def _forward(self, x):
        """forward function

        Args:
           x: input tensor of size=(batch, Cin, Hin, Win) for image or
              size=(batch, C, freq, time) for audio

        Returns:
           Tensor with output logits of size=(batch, out_units) if out_units>0,
           otherwise, it returns tensor of represeantions of size=(batch, Cout, Hout, Wout)

        """

        if self.in_norm:
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

    def get_config(self):
        """Gets network config
        Returns:
           dictionary with config params
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

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# SpineNet structures from the original paper
class SpineNet49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 256
        kwargs["filter_size_scale"] = 1.0
        kwargs["resample_alpha"] = 0.5
        kwargs["block_repeats"] = 1
        super(SpineNet49, self).__init__(in_channels, **kwargs)


class SpineNet49S(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["resample_alpha"] = 0.5
        kwargs["block_repeats"] = 1
        super(SpineNet49S, self).__init__(in_channels, **kwargs)


class SpineNet96(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 256
        kwargs["filter_size_scale"] = 1.0
        kwargs["resample_alpha"] = 0.5
        kwargs["block_repeats"] = 2
        super(SpineNet96, self).__init__(in_channels, **kwargs)


class SpineNet143(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 256
        kwargs["filter_size_scale"] = 1.0
        kwargs["resample_alpha"] = 1.0
        kwargs["block_repeats"] = 3
        super(SpineNet143, self).__init__(in_channels, **kwargs)


class SpineNet190(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 512
        kwargs["filter_size_scale"] = 1.3
        kwargs["resample_alpha"] = 1.0
        kwargs["block_repeats"] = 4
        super(SpineNet190, self).__init__(in_channels, **kwargs)


# SpineNet modifications
# Light SpineNets
class LSpineNet49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super(LSpineNet49, self).__init__(in_channels, **kwargs)


class LSpineNet49_subpixel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["upsampling_type"] = "subpixel"
        super(LSpineNet49_subpixel, self).__init__(in_channels, **kwargs)


class LSpineNet49_bilinear(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["upsampling_type"] = "bilinear"
        super(LSpineNet49_bilinear, self).__init__(in_channels, **kwargs)


class LSpineNet49_5(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["output_levels"] = [5]
        kwargs["do_endpoint_conv"] = False
        kwargs["block_specs"] = SPINENET_BLOCK_SPECS_5
        super(LSpineNet49_5, self).__init__(in_channels, **kwargs)


class LSpine2Net49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["is_res2net"] = True
        super(LSpine2Net49, self).__init__(in_channels, **kwargs)


# Spine2Nets ans(Time-)Squeeze-and-Excitation
class SELSpine2Net49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        super(SELSpine2Net49, self).__init__(in_channels, **kwargs)


class TSELSpine2Net49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        kwargs["time_se"] = True
        super(TSELSpine2Net49, self).__init__(in_channels, **kwargs)


class Spine2Net49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["is_res2net"] = True
        super(Spine2Net49, self).__init__(in_channels, **kwargs)


class SESpine2Net49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        super(SESpine2Net49, self).__init__(in_channels, **kwargs)


class TSESpine2Net49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        kwargs["time_se"] = True
        super(TSESpine2Net49, self).__init__(in_channels, **kwargs)


class Spine2Net49S(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["is_res2net"] = True
        super(Spine2Net49S, self).__init__(in_channels, **kwargs)


class SESpine2Net49S(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        super(SESpine2Net49S, self).__init__(in_channels, **kwargs)


class TSESpine2Net49S(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 128
        kwargs["filter_size_scale"] = 0.66
        kwargs["is_res2net"] = True
        kwargs["has_se"] = True
        kwargs["time_se"] = True
        super(TSESpine2Net49S, self).__init__(in_channels, **kwargs)


# R0-SP53 (structure from the paper)
class LR0_SP53(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["endpoints_num_filters"] = 64
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["block_specs"] = R0_SP53_BLOCK_SPECS
        super(LR0_SP53, self).__init__(in_channels, **kwargs)


class R0_SP53(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["block_specs"] = R0_SP53_BLOCK_SPECS
        super(R0_SP53, self).__init__(in_channels, **kwargs)


# concatenation
class SpineNet49_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["concat"] = True
        super(SpineNet49_concat_time, self).__init__(in_channels, **kwargs)
