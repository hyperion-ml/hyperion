from __future__ import absolute_import

from .spinenet import *

spinenet_dict = {
    'spinenet49': SpineNet49,
    'spinenet49s': SpineNet49S,
    'spinenet96': SpineNet96,
    'spinenet143': SpineNet143,
    'spinenet190': SpineNet190,
    'lspinenet49_subpixel': LSpineNet49_subpixel,
    'lspinenet49_128': LSpineNet49_128,
    'lspinenet49_3': LSpineNet49_3,
    'lspinenet49_4': LSpineNet49_4,
    'lspinenet49_5': LSpineNet49_5,
    'lspinenet49_6': LSpineNet49_6,
    'lspinenet49_7': LSpineNet49_7,
    'lspinenet49_nearest_concat_time': LSpineNet49_nearest_concat_time,
    'lspinenet49_nearest_concat_time_conv': LSpineNet49_nearest_concat_time_conv,
    'lspinenet49_nearest_concat_channel': LSpineNet49_nearest_concat_channel,
    'lspinenet49_nearest_concat_channel_endp': LSpineNet49_nearest_concat_channel_endp,
    'lspinenet49_nearest_concat_channel_chann_conv_256': LSpineNet49_nearest_concat_channel_chann_conv_256,
    'lspinenet49_128_concat_freq': LSpineNet49_128_concat_freq,
    'spinenet49_concat_time': SpineNet49_concat_time,
    'spinenet49_concat_channel': SpineNet49_concat_channel,
    'spinenet49_512': SpineNet49_512,
    'spinenet49_512_concat_time': SpineNet49_512_concat_time,
    'spinenet49_512_concat_channel': SpineNet49_512_concat_channel,
    'spinenet_lsp53': LSP53,
    'spinenet_lsp53_basic': LSP53_Basic,
    'lspinenet49_128_aggr': LSpineNet49_128_aggr,
    'lspinenet49_nearest': LSpineNet49_nearest,
    'lspinenet49_bilinear': LSpineNet49_bilinear,
    'spinenet49_nearest': SpineNet49_nearest,
    'lspinenet49_nearest_avg5': LSpineNet49_nearest_avg5,
    'lspinenet49_avg5': LSpineNet49_avg5,
    'lspinenet49_nearest_avg5_concat_channel': LSpineNet49_nearest_avg5_concat_channel,
    'lspinenet49_nearest_res2net': LSpineNet49_nearest_res2net,
    'lspinenet49_nearest_res2net_se': LSpineNet49_nearest_res2net_se,
    'lspinenet49_nearest_res2net_tse': LSpineNet49_nearest_res2net_tse,
    'spinenet49_nearest_res2net': SpineNet49_nearest_res2net,
    'spinenet49_nearest_res2net_se': SpineNet49_nearest_res2net_se,
    'spinenet49_nearest_se': SpineNet49_nearest_se,
    'spinenet49_nearest_res2net_tse': SpineNet49_nearest_res2net_tse,
    'lspinenet49_nearest_res2net_bn': LSpineNet49_nearest_res2net_bn,
    'lspinenet49_nearest_weighted': LSpineNet49_nearest_weighted,
    'lspinenet49_aggr_noup': LSpineNet49_aggr_noup,
    'spinenet49_aggr_noup': SpineNet49_aggr_noup,
    'spinenet49_res2net_std_se': SpineNet49_res2net_std_se,
    'lspinenet49_aggr_upfirst': LSpineNet49_aggr_upfirst,
    'lspinenet49_nearest_upfirst': LSpineNet49_nearest_upfirst,
    'spinenet49_aggr_noup_noconv': SpineNet49_aggr_noup_noconv,
    'lspinenet49_nearest_concat_channel_endp_upfirst': LSpineNet49_nearest_concat_channel_endp_upfirst,
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
               se_r=16, in_feats=None,
               res2net_scale=4, res2net_width_factor=1):
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
            se_r=se_r, in_feats=in_feats,
            res2net_scale=res2net_scale, res2net_width_factor=res2net_width_factor)

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
                      'in_norm', 'norm_layer', 'norm_before', 'do_maxpool',
                      'se_r', 'res2net_scale', 'res2net_width_factor')

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

        parser.add_argument(
            p1+'se-r', default=16, type=int,
            help=('squeeze ratio in squeeze-excitation blocks'))

        parser.add_argument(
            p1 + 'res2net-scale', default=4, type=int,
            help=('scale parameter for res2net'))

        parser.add_argument(
            p1 + 'res2net-width-factor', default=1, type=float,
            help=('multiplicative factor for the internal width of res2net'))

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

