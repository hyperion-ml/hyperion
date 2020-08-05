"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .resnet import *

resnet_dict = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'resnext50_32x4d': ResNext50_32x4d,
    'resnext101_32x8d': ResNext101_32x8d,
    'wideresnet50': WideResNet50,
    'wideresnet101': WideResNet101,
    'lresnet18': LResNet18,
    'lresnet34': LResNet34,
    'lresnet50': LResNet50,
    'lresnext50_4x4d': LResNext50_4x4d,
    'seresnet18': SEResNet18,
    'seresnet34': SEResNet34,
    'seresnet50': SEResNet50,
    'seresnet101': SEResNet101,
    'seresnet152': SEResNet152,
    'seresnext50_32x4d': SEResNext50_32x4d,
    'seresnext101_32x8d': SEResNext101_32x8d,
    'sewideresnet50': SEWideResNet50,
    'sewideresnet101': SEWideResNet101,
    'selresnet18': SELResNet18,
    'selresnet34': SELResNet34,
    'selresnet50': SELResNet50,
    'selresnext50_4x4d': SELResNext50_4x4d,
    'tseresnet18': TSEResNet18,
    'tseresnet34': TSEResNet34,
    'tseresnet50': TSEResNet50,
    'tseresnet101': TSEResNet101,
    'tseresnet152': TSEResNet152,
    'tseresnext50_32x4d': TSEResNext50_32x4d,
    'tseresnext101_32x8d': TSEResNext101_32x8d,
    'tsewideresnet50': TSEWideResNet50,
    'tsewideresnet101': TSEWideResNet101,
    'tselresnet18': TSELResNet18,
    'tselresnet34': TSELResNet34,
    'tselresnet50': TSELResNet50,
    'tselresnext50_4x4d': TSELResNext50_4x4d
}



class ResNetFactory(object):

    @staticmethod
    def create(resnet_type, in_channels, conv_channels=64, base_channels=64, out_units=0,
               hid_act={'name':'relu6', 'inplace': True}, out_act=None,
               in_kernel_size=7, in_stride=2,
               zero_init_residual=False,
               groups=1, replace_stride_with_dilation=None, dropout_rate=0,
               norm_layer=None, norm_before=True, do_maxpool=True, in_norm=True, 
               se_r=16, in_feats=None):

        try:
            resnet_class = resnet_dict[resnet_type]
        except:
            raise Exception('%s is not valid ResNet network' % (resnet_type))

        resnet = resnet_class(
            in_channels, conv_channels=conv_channels, base_channels=base_channels, 
            out_units=out_units,
            hid_act=hid_act, out_act=out_act, 
            in_kernel_size=in_kernel_size, in_stride=in_stride,
            zero_init_residual=zero_init_residual,
            groups=groups, replace_stride_with_dilation=replace_stride_with_dilation, 
            dropout_rate=dropout_rate,
            norm_layer=norm_layer, norm_before=norm_before, 
            do_maxpool=do_maxpool, in_norm=in_norm,
            se_r=se_r, in_feats=in_feats)

        return resnet


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


        valid_args = ('resnet_type', 'in_channels',
                      'conv_channels', 'base_channels', 'out_units',
                      'hid_act', 'out_act', 'in_kernel_size', 'in_stride',
                      'zero_init_residual', 'groups', 
                      'replace_stride_with_dilation', 'dropout_rate',
                      'in_norm', 'norm_layer', 'norm_before', 'do_maxpool', 
                      'se_r')

        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        return args



    @staticmethod
    def add_argparse_args(parser, prefix=None):
        
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'

        resnet_types = resnet_dict.keys()

        parser.add_argument(
            p1+'resnet-type', type=str.lower, default='lresnet34',
            choices=resnet_types, help=('ResNet type'))

        parser.add_argument(
            p1+'in-channels', default=1, type=int,
            help=('number of input channels'))

        parser.add_argument(
            p1+'conv-channels', default=64, type=int,
            help=('number of output channels in input convolution '))

        parser.add_argument(
            p1+'base-channels', default=64, type=int,
            help=('base channels of first ResNet block'))

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


