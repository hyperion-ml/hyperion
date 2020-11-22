"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .spinenet import *

spinenet_dict = {
    'spinenet49': SpineNet49,
    'spinenet49s': SpineNet49S,
    'spinenet96': SpineNet96,
    'spinenet143': SpineNet143,
    'spinenet190': SpineNet190,
    'lspinenet49': LSpineNet49,
    'lspinenet49_128': LSpineNet49_128,
    'lspinenet49_c': LSpineNet49_c,
    'lspinenet49_3': LSpineNet49_3,
    'lspinenet49_5': LSpineNet49_5,
    'lspinenet49_7': LSpineNet49_7,
    'lspinenet49_c_nonup': LSpineNet49_c_nonup,
    'lspinenet49_128_c': LSpineNet49_128_c,
    'lspinenet49_128_c_nonup': LSpineNet49_128_c_nonup,
    'lspinenet49_128_c_axf': LSpineNet49_128_C_axF,
    'lspinenet49_256': LSpineNet49_256,
    'lspinenet49_128_avgto5': LSpineNet49_128_avgto5,
    'spinenet49ss': SpineNet49SS,
}


class SpineNetFactory(object):

    @staticmethod
    def create(spinenet_type, in_channels, output_levels = [3, 4, 5, 6, 7], endpoints_num_filters = 256, resample_alpha = 0.5,
               block_repeats = 1, filter_size_scale = 1.0,  conv_channels=64, base_channels=64, out_units=0,
               hid_act={'name':'relu6', 'inplace': True}, out_act=None,
               in_kernel_size=7, in_stride=2,
               zero_init_residual=False,
               groups=1, replace_stride_with_dilation=None, dropout_rate=0,
               norm_layer=None, norm_before=True, do_maxpool=True, in_norm=True, 
               in_feats=None):
        try:
            spinenet_class = spinenet_dict[spinenet_type]
        except:
            raise Exception('%s is not valid SpineNet network' % (spinenet_type))

        spinenet = spinenet_class(
            in_channels, output_levels=output_levels, endpoints_num_filters=endpoints_num_filters,
            resample_alpha=resample_alpha, block_repeats = block_repeats, filter_size_scale = filter_size_scale,
            conv_channels=conv_channels, base_channels=base_channels, out_units=out_units,
            hid_act=hid_act, out_act=out_act, 
            in_kernel_size=in_kernel_size, in_stride=in_stride,
            zero_init_residual=zero_init_residual,
            groups=groups, replace_stride_with_dilation=replace_stride_with_dilation, 
            dropout_rate=dropout_rate,
            norm_layer=norm_layer, norm_before=norm_before, 
            do_maxpool=do_maxpool, in_norm=in_norm,
            in_feats=in_feats)

        return spinenet


    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if 'norm_after' in kwargs:
            kwargs['norm_before'] = not kwargs['norm_after']
            del kwargs['norm_after']

        if 'no_maxpool' in kwargs:
            kwargs['do_maxpool'] = not kwargs['no_maxpool']
            del kwargs['no_maxpool']


        valid_args = ('spinenet_type', 'in_channels',
                      'conv_channels', 'base_channels', 'out_units',
                      'hid_act', 'out_act', 'in_kernel_size', 'in_stride',
                      'zero_init_residual', 'groups', 
                      'replace_stride_with_dilation', 'dropout_rate',
                      'in_norm', 'norm_layer', 'norm_before', 'do_maxpool')

        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        return args



    @staticmethod
    def add_argparse_args(parser, prefix=None):
        
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'

        spinenet_types = spinenet_dict.keys()

        parser.add_argument(
            p1+'spinenet-type', type=str.lower, default='spinenet49',
            choices=spinenet_types, help=('SpineNet type'))

        parser.add_argument(
            p1+'in-channels', default=1, type=int,
            help=('number of input channels'))

        parser.add_argument(
            p1+'conv-channels', default=64, type=int,
            help=('number of output channels in input convolution '))

        parser.add_argument(
            p1+'base-channels', default=64, type=int,
            help=('base channels of first SpineNet block'))

        parser.add_argument(p1+'in-kernel-size', 
                            default=7, type=int,
                            help=('kernel size of first convolution'))

        parser.add_argument(p1+'in-stride', 
                            default=2, type=int,
                            help=('stride of first convolution'))

        parser.add_argument(p1+'groups', 
                            default=1, type=int,
                            help=('number of groups in residual blocks convolutions'))

        try:
            parser.add_argument(
                p1+'norm-layer', default=None, 
                choices=['batch-norm', 'group-norm', 'instance-norm', 'instance-norm-affine', 'layer-norm'],
                help='type of normalization layer')
        except:
            pass

        parser.add_argument(p1+'in-norm', default=False, action='store_true',
                            help='batch normalization at the input')

        parser.add_argument(p1+'no-maxpool', default=False, action='store_true',
                            help='don\'t do max pooling after first convolution')

        parser.add_argument(p1+'zero-init-residual', default=False, action='store_true',
                            help='Zero-initialize the last BN in each residual branch')

        # parser.add_argument(p1+'replace-stride-with-dilation', default=None, nargs='+', type=bool,
        #  help='replaces strides with dilations to increase context without downsampling')

        # parser.add_argument(
        #     p1+'se-r', default=16, type=int,
        #     help=('squeeze ratio in squeeze-excitation blocks'))

        try:
            parser.add_argument(p1+'hid-act', default='relu6', 
                                help='hidden activation')
        except:
            pass
        
        try:
            parser.add_argument(p1+'norm-after', default=False, action='store_true',
                                help='batch normalizaton after activation')
        except:
            pass
        
        try:
            parser.add_argument(p1+'dropout-rate', default=0, type=float,
                                help='dropout')
        except:
            pass

