import numpy as np

import logging

import torch
import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from ..layer_blocks import ResNetInputBlock, ResNetBasicBlock, ResNetBNBlock
from ..layer_blocks import BlockSpec, SpineResample, SpineEndpoints
from .net_arch import NetArch


SPINENET_BLOCK_SPECS = [
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

FILTER_SIZE_MAP = {
    1: 0.5,
    2: 1,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
}


class SpineNet(NetArch):

    def __init__(self, in_channels, block_specs = None, output_levels=[3, 4, 5, 6, 7], endpoints_num_filters=256,
                 resample_alpha=0.5, concat=False, do_endpoint_conv=True, concat_ax=3,
                 feature_output_level=None,
                 block_repeats=1, filter_size_scale=1.0, conv_channels=64, base_channels=64, out_units=0,
                 hid_act={'name':'relu6', 'inplace': True}, out_act=None,
                 in_kernel_size=7, in_stride=2,
                 zero_init_residual=False,
                 groups=1, replace_stride_with_dilation=None, dropout_rate=0,
                 norm_layer=None, norm_before=True, do_maxpool=True, in_norm=True,
                 in_feats=None):

        super(SpineNet, self).__init__()
        self.register_buffer('step', torch.tensor(0.))

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
        self._block_specs = BlockSpec.build_block_specs(SPINENET_BLOCK_SPECS) \
            if block_specs is None else BlockSpec.build_block_specs(block_specs)
        self.output_levels = output_levels
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
        if norm_layer == 'group-norm':
            norm_groups = min(base_channels//2, 32)
            norm_groups = max(norm_groups, groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        if in_norm:
            self.in_bn = norm_layer(in_channels)

        self.in_block = ResNetInputBlock(
            in_channels, conv_channels, kernel_size=in_kernel_size, stride=in_stride,
            activation=hid_act, norm_layer=self._norm_layer, norm_before=norm_before, do_maxpool=do_maxpool)

        # self._context = self.in_block.context
        # self._downsample_factor = self.in_block.downsample_factor
        self.cur_in_channels = conv_channels

        self.stem0 = self._make_layer(ResNetBNBlock, 2, self.block_repeats, in_channels=conv_channels)
        self.stem1 = self._make_layer(ResNetBNBlock, 2, self.block_repeats)

        self.stem_nbr = 2
        self.blocks = self._make_permuted_blocks(self._block_specs[self.stem_nbr:])
        self.connections = self._make_permuted_connections(self._block_specs[self.stem_nbr:])

        self.endpoints = self._make_endpoints()

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
        blocks = nn.ModuleList([])
        for block in block_specs:
            layer_i = self._make_layer(block.block_fn, block.level, self.block_repeats)
            blocks.append(layer_i)
        return blocks

    def _make_permuted_connections(self, block_specs):
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
                                                   scale, self.resample_alpha, activation=self.hid_act,
                                                   norm_layer=self._norm_layer, norm_before=self.norm_before))
            connections_i.append(AF.create(self.hid_act))
            connections.append(connections_i)
        return connections

    def _make_endpoints(self):
        endpoints = nn.ModuleDict()
        for block_spec in self._block_specs:
            if block_spec.is_output and block_spec.level in self.output_levels:
                in_channels = int(FILTER_SIZE_MAP[block_spec.level] * self.filter_size_scale * self.base_channels) * 4
                endpoints[str(block_spec.level)] = SpineEndpoints(in_channels, self.endpoints_num_filters,
                                                                  block_spec.level, self.feature_output_level,
                                                                  activation=self.hid_act,
                                                                  norm_layer=self._norm_layer,
                                                                  norm_before=self.norm_before,
                                                                  do_endpoint_conv=self.do_endpoint_conv)
        return endpoints

    def _make_layer(self, block, block_level, num_blocks, in_channels=None, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        kwargs = {}
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
        if not self.do_endpoint_conv:
            C = 0
            for output_level in self.output_levels:
                C += self.base_channels * 4 * FILTER_SIZE_MAP[output_level]
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
            if self.concat_ax == 2 and self.concat:
                H = H*len(self.output_levels)

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
        # match shape function for feats inside permuted network
        surplus = feat1.size(2) - feat1.size(2)
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
            input0 = block.input_offsets[0]
            input1 = block.input_offsets[1]

            parent0_feat = self.connections[idx][0](feats[input0])
            parent1_feat = self.connections[idx][1](feats[input1])
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

            target_feat = self.connections[idx][2](target_feat)
            x = self.blocks[idx](target_feat)

            feats.append(x)
            num_outgoing_connections.append(0)
            if block.is_output and block.level in self.output_levels:
                if str(block.level) in output_feats:
                    raise ValueError(
                        'Duplicate feats found for output level {}.'.format(block.level))
                output_feats[str(block.level)] = x

        output_endpoints = []
        output_shape = list(output_feats[str(self.feature_output_level)].size())
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
                  'in_feats': self.in_feats
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


class SpineNet49_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['concat'] = True
        super(SpineNet49_concat_time, self).__init__(
            in_channels, **kwargs)


class SpineNet49_concat_channel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        super(SpineNet49_concat_channel, self).__init__(
            in_channels, **kwargs)


class SpineNet49S(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 0.65
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        super(SpineNet49S, self).__init__(
            in_channels, **kwargs)


class SpineNet49SS(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 0.25
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        super(SpineNet49SS, self).__init__(
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


# Light SpineNets
class LSpineNet49(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        super(LSpineNet49, self).__init__(
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
        super(LSpineNet49_3, self).__init__(
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
        super(LSpineNet49_5, self).__init__(
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
        super(LSpineNet49_7, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 64
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        super(LSpineNet49_concat_time, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_concat_channel(SpineNet):
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
        super(LSpineNet49_concat_channel, self).__init__(
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

class LSpineNet49_128_avgto5(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['feature_output_level'] = 5
        super(LSpineNet49_128_avgto5, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_128_concat_time(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        super(LSpineNet49_128_concat_time, self).__init__(
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
        kwargs['do_endpoint_conv'] = True
        kwargs['concat_ax'] = 2
        super(LSpineNet49_128_concat_freq, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_128_concat_channel(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 128
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        kwargs['concat'] = True
        kwargs['do_endpoint_conv'] = False
        kwargs['concat_ax'] = 1
        super(LSpineNet49_128_concat_channel, self).__init__(
            in_channels, **kwargs)

class LSpineNet49_256(SpineNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['endpoints_num_filters'] = 256
        kwargs['filter_size_scale'] = 1.0
        kwargs['resample_alpha'] = 0.5
        kwargs['block_repeats'] = 1
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        super(LSpineNet49_256, self).__init__(
            in_channels, **kwargs)