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
    (4, ResNetBNBlock, (8, 10), False), # 11
    (3, ResNetBNBlock, (4, 10), True),  # 12
    (4, ResNetBNBlock, (6, 7), True),   # 13 it has 3 inputs
    (5, ResNetBNBlock, (8, 13), True),  # 14
    (7, ResNetBNBlock, (6, 9), True),   # 15
    (6, ResNetBNBlock, (7, 9), True),   # 16
]

SPINENET_BLOCK_SPECS_5 = [
    # level, block type, tuple of inputs, is output
    (2, ResNetBNBlock, (None, None), False), # 0
    (2, ResNetBNBlock, (None, None), False), # 1
    (2, ResNetBNBlock, (0, 1), False), # 2
    (4, ResNetBasicBlock, (0, 1), False), # 3
    (3, ResNetBNBlock, (2, 3), False), # 4
    (4, ResNetBNBlock, (2, 4), False), # 5
    (6, ResNetBasicBlock, (3, 5), False), # 6
    (4, ResNetBNBlock, (3, 5), False), # 7
    (5, ResNetBasicBlock, (6, 7), False), # 8
    (7, ResNetBasicBlock, (6, 8), False), # 9
    (5, ResNetBNBlock, (8, 9), False), # 10
    (5, ResNetBNBlock, (8, 10), False), # 11
    (4, ResNetBNBlock, (5, 10), True), # 12
    # (3, ResNetBNBlock, (4, 10), True), # 13
    (5, ResNetBNBlock, (7, 12), True), # 14
    # (7, ResNetBNBlock, (5, 14), True), # 15
    # (6, ResNetBNBlock, (12, 14), True), # 16
]

SPINENET_BLOCK_SPECS_345 = [
    # level, block type, tuple of inputs, is output
    (2, ResNetBNBlock, (None, None), False), # 0
    (2, ResNetBNBlock, (None, None), False), # 1
    (2, ResNetBNBlock, (0, 1), False), # 2
    (4, ResNetBasicBlock, (0, 1), False), # 3
    (3, ResNetBNBlock, (2, 3), False), # 4
    (4, ResNetBNBlock, (2, 4), False), # 5
    (6, ResNetBasicBlock, (3, 5), False), # 6
    (4, ResNetBNBlock, (3, 5), False), # 7
    (5, ResNetBasicBlock, (6, 7), False), # 8
    (7, ResNetBasicBlock, (6, 8), False), # 9
    (5, ResNetBNBlock, (8, 9), False), # 10
    (5, ResNetBNBlock, (8, 10), False), # 11
    (4, ResNetBNBlock, (5, 10), True), # 12
    (3, ResNetBNBlock, (4, 10), True), # 13
    (5, ResNetBNBlock, (7, 12), True), # 14
    # (7, ResNetBNBlock, (5, 14), True), # 15
    # (6, ResNetBNBlock, (12, 14), True), # 16
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

    def __init__(self, in_channels, block_specs=None, output_levels=[3, 4, 5, 6, 7], endpoints_num_filters=256,
                 resample_alpha=0.5, concat=False, do_endpoint_conv=True, concat_ax=3, upsampling_type='nearest',
                 feature_output_level=None, aggregation=False, time_conv=False, channel_conv=False, chann_conv_chann=None,
                 weighted_sum=False, do_endpoint_upsampling=True, end_upsample_before=False,
                 endpoint_sum=False, do_endpoint_relu=False,
                 block_repeats=1, filter_size_scale=1.0, conv_channels=64, base_channels=64, out_units=0,
                 hid_act={'name':'relu6', 'inplace': True}, out_act=None,
                 in_kernel_size=7, in_stride=2,
                 zero_init_residual=False,
                 groups=1, replace_stride_with_dilation=None, dropout_rate=0,
                 norm_layer=None, norm_before=True, do_maxpool=True, in_norm=True,
                 in_feats=None, se_r=16, time_se=False, has_se = False, std_se=False,
                 is_res2net=False, res2net_scale=4, res2net_width_factor=1, res2net_basic=True):
        """
        Base class for the SpineNet structure.
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
        super(SpineNet, self).__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.base_channels = base_channels
        self.out_units = out_units
        self.endpoints_num_filters = endpoints_num_filters
        self.resample_alpha = resample_alpha
        self.block_repeats = block_repeats
        self.filter_size_scale = filter_size_scale
        self.concat = concat
        self.do_endpoint_conv = do_endpoint_conv
        self.concat_ax = concat_ax
        self.feature_output_level = min(output_levels) if feature_output_level is None else feature_output_level
        self.aggregation = aggregation
        self.time_conv = time_conv
        self.channel_conv = channel_conv
        self.chann_conv_chann = chann_conv_chann
        self.do_endpoint_upsampling = do_endpoint_upsampling
        self.end_upsample_before = end_upsample_before
        self.endpoint_sum  = endpoint_sum
        self.do_endpoint_relu = do_endpoint_relu

        self.res2net_scale = res2net_scale
        self.res2net_width_factor = res2net_width_factor
        self.is_res2net = is_res2net
        self.res2net_basic = res2net_basic

        self.se_r=se_r
        self.time_se=time_se
        self.has_se = has_se
        self.std_se = std_se

        self.weighted_sum = weighted_sum

        self._block_specs = BlockSpec.build_block_specs(SPINENET_BLOCK_SPECS) \
            if block_specs is None else BlockSpec.build_block_specs(block_specs)
        self.output_levels = output_levels
        self.upsampling_type = upsampling_type
        self.dilation = 1
        # this also need testing
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.replace_stride_with_dilation = replace_stride_with_dilation

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
        if norm_layer == 'group-norm':
            norm_groups = min(base_channels//2, 32)
            norm_groups = max(norm_groups, groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        if in_norm:
            self.in_bn = norm_layer(in_channels)

        self.in_block = ResNetInputBlock(
            in_channels, conv_channels, kernel_size=in_kernel_size, stride=in_stride,
            activation=hid_act, norm_layer=self._norm_layer, norm_before=norm_before, do_maxpool=do_maxpool)

        # TO DO make these useful
        # self._context = self.in_block.context
        # self._downsample_factor = self.in_block.downsample_factor
        self.cur_in_channels = conv_channels

        if self.is_res2net:
            if self._block_specs[0].block_fn == ResNetBNBlock:
                _in_block = Res2NetBNBlock
            elif block.block_fn == ResNetBasicBlock and self.res2net_basic:
                _in_block = Res2NetBasicBlock
            elif block.block_fn == ResNetBasicBlock and not self.res2net_basic:
                _in_block = ResNetBasicBlock
        else:
            _in_block = self._block_specs[0].block_fn

        self.stem0 = self._make_layer(_in_block, 2, self.block_repeats, in_channels=conv_channels)
        self.stem1 = self._make_layer(_in_block, 2, self.block_repeats)

        self.stem_nbr = 2  # the number of the stem layers
        self.blocks = self._make_permuted_blocks(self._block_specs[self.stem_nbr:])
        self.connections = self._make_permuted_connections(self._block_specs[self.stem_nbr:])

        self.endpoints = self._make_endpoints()

        if self.time_conv:
            self.endpoint_conv = SpineConv(self.endpoints_num_filters, self.endpoints_num_filters)
        elif self.channel_conv:
            in_chann_conv = 0
            for idx in self.output_levels:
                in_chann_conv += self.base_channels * 4 * FILTER_SIZE_MAP[idx]
            self.endpoint_conv = SpineConv(in_chann_conv, self.chann_conv_chann)

        if self.weighted_sum:
            self.weight_endpoints = nn.Parameter(torch.ones((len(output_levels),)))

        if self.do_endpoint_relu:
            self.endpoint_relu = AF.create(self.hid_act)

        self.with_output = False
        self.out_act = None
        if out_units > 0:
            self.with_output = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output = nn.Linear(self.cur_in_channels, out_units)
            self.out_act = AF.create(out_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = 'relu'
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act['name']
                if act_name == 'swish':
                    act_name = 'relu'
                try:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=act_name)
                except:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
                elif block.block_fn == ResNetBasicBlock and self.res2net_basic:
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
            out_channels = int(FILTER_SIZE_MAP[block.level] * self.filter_size_scale * self.base_channels) * expansion

            connections_i = nn.ModuleList([])
            for i in block.input_offsets:
                offset_block = self._block_specs[i]
                scale = offset_block.level - block.level
                in_channels = int(FILTER_SIZE_MAP[offset_block.level] * self.filter_size_scale * self.base_channels)
                connections_i.append(SpineResample(offset_block, in_channels, out_channels,
                                                   scale, self.resample_alpha, self.upsampling_type, activation=self.hid_act,
                                                   norm_layer=self._norm_layer, norm_before=self.norm_before))
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
                in_channels = int(FILTER_SIZE_MAP[block_spec.level] * self.filter_size_scale * self.base_channels) * expansion
                out_channels = self.endpoints_num_filters if self.do_endpoint_conv else in_channels
                endpoints[str(block_spec.level)] = SpineEndpoints(in_channels, out_channels,
                                                                      block_spec.level, self.feature_output_level,
                                                                      self.upsampling_type,
                                                                      activation=self.hid_act,
                                                                      norm_layer=self._norm_layer,
                                                                      norm_before=self.norm_before,
                                                                      do_endpoint_conv=self.do_endpoint_conv,
                                                                      do_endpoint_upsampling=self.do_endpoint_upsampling,
                                                                      end_upsample_before=self.end_upsample_before)
        return endpoints

    def _make_layer(self, block, block_level, num_blocks, in_channels=None, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        kwargs = {}

        # To DO make it more generalized
        if self.has_se:
            if self.time_se:
                num_feats = self.in_feats
                for i in range(block_level-2):
                    num_feats = int(num_feats // 2) if num_feats % 2 == 0 else int(num_feats // 2 + 1)
                # num_feats = int(self.in_feats/2**(block_level-2))
                # logging.info(num_feats)
                kwargs = {'se_r': self.se_r, 'time_se': True, 'num_feats': num_feats}
            elif self.std_se:
                kwargs = {'std_se': self.std_se, 'se_r': self.se_r}
            else:
                kwargs = {'se_r': self.se_r}

        if self.is_res2net and block != ResNetBasicBlock:
            kwargs['scale'] = self.res2net_scale
            kwargs['width_factor'] = self.res2net_width_factor
        channels = int(FILTER_SIZE_MAP[block_level] * self.base_channels * self.filter_size_scale)
        if in_channels is None:
            in_channels = channels * block.expansion
        layers = []

        layers.append(block(
            in_channels, channels, activation=self.hid_act,
            stride=stride, dropout_rate=self.dropout_rate, groups=self.groups,
            dilation=previous_dilation,
            norm_layer=self._norm_layer, norm_before=self.norm_before, **kwargs))

        # self._context += layers[0].context * self._downsample_factor
        # self._downsample_factor *= layers[0].downsample_factor

        for _ in range(1, num_blocks):
            layers.append(block(
                in_channels, channels, activation=self.hid_act,
                dropout_rate=self.dropout_rate,
                groups=self.groups, dilation=self.dilation,
                norm_layer=self._norm_layer, norm_before=self.norm_before, **kwargs))
            # self._context += layers[-1].context * self._downsample_factor

        return nn.Sequential(*layers)

    def _compute_out_size(self, in_size):
        """Computes output size given input size.
           Output size is not the same as input size because of
           downsampling steps.

        Args:
           in_size: input size of the H or W dimensions

        Returns:
           output_size
        """
        out_size = int((in_size - 1)//self.in_stride+1)
        if self.do_maxpool:
            out_size = int((out_size - 1)//2+1)

        downsample_levels = self.feature_output_level - 2
        for i in range(downsample_levels):
            out_size = int(out_size//2) if out_size % 2 == 0 else int(out_size//2+1)

        return out_size

    def _compute_channel_size(self):
        """
        Returns:
          If the 1x1 conv is not conducted in the endpoint blocks, the number of channels is equal to the sum of the
          nbr of channels of the output blocks.
        """
        if self.channel_conv:
            return self.chann_conv_chann
        if not self.do_endpoint_conv:
            C = 0
            for output_level in self.output_levels:
                C += self.base_channels * 4 * FILTER_SIZE_MAP[output_level]
            return C
        else:
            if self.concat and self.concat_ax==1:
                C = len(self.output_levels)*self.endpoints_num_filters
                return C
        return self.endpoints_num_filters

    # TO DO
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
    #     """

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
                H = H*len(self.output_levels)

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])

        C = self._compute_channel_size()
        if self.aggregation:
            return (len(self.output_levels), in_shape[0], C, H, W)
        # if self.aggregation:
        #     if not self.do_endpoint_conv and not self.do_endpoint_upsampling:
        #         # return (in_shape[0], 7424, 2, W)
        #         return (in_shape[0], 29696, 2, W)
        #     elif self.do_endpoint_conv and not self.do_endpoint_upsampling:
        #         return (in_shape[0], 2496, 2, W)
        #         # return (in_shape[0], 9984, 2, W)
        #     else:
        #         # return (in_shape[0], len(self.output_levels)*C, H, W)
        #         return (len(self.output_levels), in_shape[0], C, H, W)
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
        for idx, block in enumerate(self._block_specs[self.stem_nbr:]):
            # logging.info(idx)
            input0 = block.input_offsets[0]
            input1 = block.input_offsets[1]

            parent0_feat = self.connections[idx][0](feats[input0])
            parent1_feat = self.connections[idx][1](feats[input1])
            # logging.info(parent0_feat.shape)
            # logging.info(parent1_feat.shape)
            parent0_feat, parent1_feat = self._match_feat_shape(parent0_feat, parent1_feat)
            target_feat = parent0_feat + parent1_feat

            num_outgoing_connections[input0] += 1
            num_outgoing_connections[input1] += 1
            # Connect intermediate blocks with outdegree 0 to the output block.
            if block.is_output:
                for j, (j_feat, j_connections) in enumerate(zip(feats, num_outgoing_connections)):
                    if j_connections == 0 and j_feat.shape == target_feat.shape:
                        target_feat += j_feat
                        num_outgoing_connections[j] += 1

            target_feat = self.connections[idx][2](target_feat)  # pass input through the activation function
            x = self.blocks[idx](target_feat)

            feats.append(x)
            num_outgoing_connections.append(0)
            if block.is_output and block.level in self.output_levels:
                if str(block.level) in output_feats:
                    raise ValueError('Duplicate feats found for output level {}.'.format(block.level))
                output_feats[str(block.level)] = x

        if self.do_endpoint_upsampling:
            output_endpoints = []
        else:
            output_endpoints = {}
        output_shape = list(output_feats[str(self.feature_output_level)].size())  # get the target output size

        for endpoint in self.endpoints:
            if self.endpoints[endpoint] is not None:
                endpoint_i = self.endpoints[endpoint](output_feats[endpoint])
            else:
                endpoint_i = output_feats[endpoint]
            if self.do_endpoint_upsampling:
                endpoint_i = self._match_shape(endpoint_i, output_shape)
                output_endpoints.append(endpoint_i)
            else:
                output_endpoints[endpoint] = endpoint_i

        if self.aggregation:
            if self.do_endpoint_upsampling:
                return torch.stack(output_endpoints)
            else:
                return output_endpoints

        if self.concat:
            x = torch.cat(output_endpoints, self.concat_ax)
            if self.time_conv or self.channel_conv:
                x = self.endpoint_conv(x)
        else:
            if self.weighted_sum:
                # x = torch.mean(self.weight_endpoints * torch.stack(output_endpoints, dim=-1), -1)
                # edge_weights = nn.functional.relu(self.weight_endpoints.to(dtype=dtype))
                # logging.info(x.shape)
                logging.info(self.weight_endpoints)
                weight_endpoints = nn.functional.relu(self.weight_endpoints)
                weights_sum = torch.sum(weight_endpoints)
                x = torch.stack(
                    [(output_endpoints[i] * weight_endpoints[i]) / (weights_sum + 0.0001) for i in range(len(output_endpoints))], dim=-1)
                x = torch.sum(x, dim=-1)
            elif self.endpoint_sum:
                x = torch.sum(torch.stack(output_endpoints), 0)
                if self.do_endpoint_relu:
                    x = self.endpoint_relu(x)
                    # logging.info(x.shape)
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
        """ Gets network config
        Returns:
           dictionary with config params
        """

        out_act = AF.get_config(self.out_act)
        hid_act = self.hid_act

        config = {'endpoints_num_filters': self.endpoints_num_filters,
                  'resample_alpha': self.resample_alpha,
                  'block_repeats': self.block_repeats,
                  'filter_size_scale': self.filter_size_scale,
                  'output_levels': self.output_levels,
                  'in_channels': self.in_channels,
                  'conv_channels': self.conv_channels,
                  'base_channels': self.base_channels,
                  'out_units': self.out_units,
                  'in_kernel_size': self.in_kernel_size,
                  'in_stride': self.in_stride,
                  'zero_init_residual': self.zero_init_residual,
                  'groups': self.groups,
                  'dropout_rate': self.dropout_rate,
                  'norm_layer': self.norm_layer,
                  'norm_before': self.norm_before,
                  'in_norm': self.in_norm,
                  'do_maxpool': self.do_maxpool,
                  'out_act': out_act,
                  'hid_act': hid_act,
                  'se_r': self.se_r,
                  'in_feats': self.in_feats,
                  'res2net_scale': self.res2net_scale,
                  'res2net_width_factor': self.res2net_width_factor
                  }

        base_config = super(SpineNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# SpineNets from the paper
class SpineNet49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        super(SpineNet49, self).__init__(
            in_channels, **kwargs)


class SpineNet49S(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 0.65
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        super(SpineNet49S, self).__init__(
            in_channels, **kwargs)


class SpineNet96(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 2
        super(SpineNet96, self).__init__(
            in_channels, **kwargs)


class SpineNet143(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 1.0
        kwargs['block_repeats'] = 3
        super(SpineNet143, self).__init__(
            in_channels, **kwargs)


class SpineNet190(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 512
        kwargs['filter_size_scale'] = 1.3
        kwargs['resample_alpha'] = 1.0
        kwargs['block_repeats'] = 4
        super(SpineNet190, self).__init__(
            in_channels, **kwargs)


# Our modification to the SpineNets
class SpineNet49_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['concat'] = True
        super(SpineNet49_concat_time, self).__init__(
            in_channels, **kwargs)


class SpineNet49_concat_channel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        super(SpineNet49_concat_channel, self).__init__(
            in_channels, **kwargs)

class SpineNet49_concat_channel_endp(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = True
        kwargs['concat_ax'] = 1
        super(SpineNet49_concat_channel_endp, self).__init__(
            in_channels, **kwargs)


class SpineNet49_512(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 512
        super(SpineNet49_512, self).__init__(
            in_channels, **kwargs)


class SpineNet49_512_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 512
        kwargs['concat'] = True
        super(SpineNet49_512_concat_time, self).__init__(
            in_channels, **kwargs)


class SpineNet49_512_concat_channel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 512
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        super(SpineNet49_512_concat_channel, self).__init__(
            in_channels, **kwargs)


# Light SpineNets
class LSpineNet49_subpixel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'subpixel'
        super(LSpineNet49_subpixel, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_nearest, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_upfirst(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['end_upsample_before'] = True
        super(LSpineNet49_nearest_upfirst, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_weighted(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['weighted_sum'] = True
        super(LSpineNet49_nearest_weighted, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_sum(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['endpoint_sum'] = True
        kwargs['do_endpoint_relu'] = False
        super(LSpineNet49_nearest_sum, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_sum_relu(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['endpoint_sum'] = True
        kwargs['do_endpoint_relu'] = True
        super(LSpineNet49_nearest_sum_relu, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_bilinear(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'bilinear'
        super(LSpineNet49_bilinear, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_avg5(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['feature_output_level'] = 5
        super(LSpineNet49_nearest_avg5, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_avg5(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'subpixel'
        kwargs['feature_output_level'] = 5
        super(LSpineNet49_avg5, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_avg5_concat_channel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['feature_output_level'] = 5
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        super(LSpineNet49_nearest_avg5_concat_channel, self).__init__(
            in_channels, **kwargs)

class SpineNet49_nearest(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['upsampling_type'] = 'nearest'
        super(SpineNet49_nearest, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_3(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [3]
        kwargs['do_endpoint_conv'] = False
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_3, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_4(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [4]
        kwargs['do_endpoint_conv'] = False
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_4, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_5(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [5]
        kwargs['do_endpoint_conv'] = False
        kwargs['upsampling_type'] = 'nearest'
        kwargs['block_specs'] = SPINENET_BLOCK_SPECS_5
        super(LSpineNet49_5, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_subpixel_345(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [3, 4, 5]
        kwargs['upsampling_type'] = 'subpixel'
        kwargs['block_specs'] = SPINENET_BLOCK_SPECS_345
        super(LSpineNet49_subpixel_345, self).__init__(
            in_channels, **kwargs)

class SpineNet49_nearest_345(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['output_levels'] = [3, 4, 5]
        kwargs['upsampling_type'] = 'nearest'
        kwargs['block_specs'] = SPINENET_BLOCK_SPECS_345
        super(SpineNet49_nearest_345, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_5_64(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [5]
        kwargs['do_endpoint_conv'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['block_specs'] = SPINENET_BLOCK_SPECS_5

        super(LSpineNet49_5_64, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_5_128(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [5]
        kwargs['do_endpoint_conv'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['block_specs'] = SPINENET_BLOCK_SPECS_5

        super(LSpineNet49_5_128, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_6(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [6]
        kwargs['do_endpoint_conv'] = False
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_6, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_7(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [7]
        kwargs['do_endpoint_conv'] = False
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_7, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_nearest_concat_time, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_nearest_concat_time_conv(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['time_conv'] = True
        super(LSpineNet49_nearest_concat_time_conv, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_concat_channel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_nearest_concat_channel, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_concat_channel_chann_conv_256(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        kwargs['upsampling_type'] = 'nearest'
        kwargs['channel_conv'] = True
        kwargs['chann_conv_chann'] = 256
        super(LSpineNet49_nearest_concat_channel_chann_conv_256, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_concat_channel_chann_conv_128(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        kwargs['upsampling_type'] = 'nearest'
        kwargs['channel_conv'] = True
        kwargs['chann_conv_chann'] = 128
        super(LSpineNet49_nearest_concat_channel_chann_conv_128, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_concat_channel_chann_conv_64(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        kwargs['upsampling_type'] = 'nearest'
        kwargs['channel_conv'] = True
        kwargs['chann_conv_chann'] = 128
        super(LSpineNet49_nearest_concat_channel_chann_conv_64, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_nearest_concat_channel_endp(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['concat_ax'] = 1
        kwargs['upsampling_type'] = 'nearest'
        super(LSpineNet49_nearest_concat_channel_endp, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_concat_channel_endp_upfirst(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['concat_ax'] = 1
        kwargs['upsampling_type'] = 'nearest'
        kwargs['end_upsample_before'] = True
        super(LSpineNet49_nearest_concat_channel_endp_upfirst, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_128(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        super(LSpineNet49_128, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_128_aggr(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['aggregation'] = True
        super(LSpineNet49_128_aggr, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_128_concat_freq(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['concat_ax'] = 2
        super(LSpineNet49_128_concat_freq, self).__init__(
            in_channels, **kwargs)

R0_SP53_BASIC_BLOCK_SPECS = [
    # level, block type, tuple of inputs, is output
    (2, ResNetBasicBlock, (None, None), False),  # 0
    (2, ResNetBasicBlock, (None, None), False),  # 1
    (2, ResNetBasicBlock, (0, 1), False),  # 2
    (3, ResNetBasicBlock, (0, 1), False),  # 3
    (3, ResNetBasicBlock, (2, 3), False),  # 4
    (4, ResNetBasicBlock, (2, 4), False),  # 5
    (4, ResNetBasicBlock, (3, 5), False),  # 6
    (3, ResNetBasicBlock, (5, 6), False),  # 7
    (5, ResNetBasicBlock, (4, 7), False),  # 8
    (4, ResNetBasicBlock, (4, 8), False),  # 9
    (4, ResNetBasicBlock, (8, 9), False),  # 10
    (4, ResNetBasicBlock, (8, 10), False), # 11
    (3, ResNetBasicBlock, (4, 10), True),  # 12
    (4, ResNetBasicBlock, (6, 7), True),   # 13
    (5, ResNetBasicBlock, (8, 13), True),  # 14
    (7, ResNetBasicBlock, (6, 9), True),   # 15
    (6, ResNetBasicBlock, (7, 9), True),   # 16
]

class LSP53_Basic(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['block_specs'] = R0_SP53_BASIC_BLOCK_SPECS
        kwargs['upsampling_type'] = 'nearest'
        super(LSP53_Basic, self).__init__(
            in_channels, **kwargs)

class LSP53(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['block_specs'] = R0_SP53_BLOCK_SPECS
        kwargs['upsampling_type'] = 'nearest'
        super(LSP53, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_nearest_res2net(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['is_res2net'] = True
        super(LSpineNet49_nearest_res2net, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_res2net_bn(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['is_res2net'] = True
        kwargs['res2net_basic'] = False

        super(LSpineNet49_nearest_res2net_bn, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_nearest_res2net_se(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['is_res2net'] = True
        kwargs['has_se'] = True
        super(LSpineNet49_nearest_res2net_se, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_nearest_res2net_tse(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['upsampling_type'] = 'nearest'
        kwargs['is_res2net'] = True
        kwargs['has_se'] = True
        kwargs['time_se'] = True
        super(LSpineNet49_nearest_res2net_tse, self).__init__(
            in_channels, **kwargs)


class LSpineNet49_aggr_noup(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['aggregation'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['do_endpoint_upsampling'] = False
        super(LSpineNet49_aggr_noup, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_aggr_upfirst(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['aggregation'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['end_upsample_before'] = True
        super(LSpineNet49_aggr_upfirst, self).__init__(
            in_channels, **kwargs)


class SpineNet49_aggr_noup(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['aggregation'] = True
        kwargs['do_endpoint_upsampling'] = False
        super(SpineNet49_aggr_noup, self).__init__(
            in_channels, **kwargs)

class SpineNet49_aggr_noup_noconv(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['aggregation'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['do_endpoint_conv'] = False
        kwargs['do_endpoint_upsampling'] = False
        super(SpineNet49_aggr_noup_noconv, self).__init__(
            in_channels, **kwargs)


class SpineNet49_res2net_std_se(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['resample_alpha'] = 0.5
        kwargs['is_res2net'] = True
        kwargs['has_se'] = True
        kwargs['std_se'] = True
        kwargs['upsampling_type'] = 'nearest'
        super(SpineNet49_res2net_std_se, self).__init__(
            in_channels, **kwargs)


class SpineNet49_nearest_res2net_se(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['is_res2net'] = True
        kwargs['upsampling_type'] = 'nearest'
        kwargs['has_se'] = True
        super(SpineNet49_nearest_res2net_se, self).__init__(
            in_channels, **kwargs)

class SpineNet49_nearest_se(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['has_se'] = True
        super(SpineNet49_nearest_se, self).__init__(
            in_channels, **kwargs)

class SpineNet49_nearest_res2net(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['is_res2net'] = True
        kwargs['upsampling_type'] = 'nearest'
        super(SpineNet49_nearest_res2net, self).__init__(
            in_channels, **kwargs)

class SpineNet49_nearest_res2net_tse(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['is_res2net'] = True
        kwargs['has_se'] = True
        kwargs['time_se'] = True
        super(SpineNet49_nearest_res2net_tse, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_345_128(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['output_levels'] = [3, 4, 5]
        kwargs['upsampling_type'] = 'nearest'
        kwargs['block_specs'] = SPINENET_BLOCK_SPECS_345
        super(LSpineNet49_345_128, self).__init__(
            in_channels, **kwargs)

class SpineNet49S_res2net(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 0.65
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['is_res2net'] = True
        super(SpineNet49S_res2net, self).__init__(
            in_channels, **kwargs)


class SpineNet49S_res2net_se(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 0.65
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['is_res2net'] = True
        kwargs['has_se'] = True
        super(SpineNet49S_res2net_se, self).__init__(
            in_channels, **kwargs)


class SpineNet49S_res2net_tse(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 0.65
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['is_res2net'] = True
        kwargs['has_se'] = True
        kwargs['time_se'] = True
        super(SpineNet49S_res2net_tse, self).__init__(
            in_channels, **kwargs)

