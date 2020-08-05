"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .tdnn import TDNNV1
from .etdnn import ETDNNV1
from .resetdnn import ResETDNNV1

class TDNNFactory(object):

    @staticmethod
    def create(tdnn_type, num_enc_blocks, 
               in_feats, enc_hid_units, enc_expand_units=None, 
               kernel_size=3, dilation=1, dilation_factor=1,
               hid_act={'name':'relu6', 'inplace':True}, 
               out_units=0, out_act=None,
               dropout_rate=0,
               norm_layer=None, use_norm=True, norm_before=True, in_norm=True):

        if enc_expand_units is not None and isinstance(enc_hid_units, int):
            if tdnn_type != 'resetdnn' :
                enc_hid_units = (num_enc_blocks - 1)*[enc_hid_units] + [enc_expand_units]

        if tdnn_type == 'tdnn':
            nnet = TDNNV1(
                num_enc_blocks, in_feats, enc_hid_units, out_units=out_units,
                kernel_size=kernel_size, 
                dilation=dilation, dilation_factor=dilation_factor,
                hid_act=hid_act, out_act=out_act, dropout_rate=dropout_rate,
                norm_layer=norm_layer, use_norm=use_norm, 
                norm_before=norm_before, in_norm=in_norm)
        elif tdnn_type == 'etdnn':
            nnet = ETDNNV1(
                num_enc_blocks, in_feats, enc_hid_units, out_units=out_units,
                kernel_size=kernel_size, 
                dilation=dilation, dilation_factor=dilation_factor,
                hid_act=hid_act,out_act=out_act, dropout_rate=dropout_rate,
                norm_layer=norm_layer, use_norm=use_norm, 
                norm_before=norm_before, in_norm=in_norm)
        elif tdnn_type == 'resetdnn':
            if enc_expand_units is None:
                enc_expand_units = enc_hid_units

            nnet = ResETDNNV1(
                num_enc_blocks, in_feats, enc_hid_units, enc_expand_units,
                out_units=out_units, kernel_size=kernel_size, 
                dilation=dilation, dilation_factor=dilation_factor,
                hid_act=hid_act, out_act=out_act, dropout_rate=dropout_rate,
                norm_layer=norm_layer, use_norm=use_norm, 
                norm_before=norm_before, in_norm=in_norm)
        else:
            raise Exception('%s is not valid TDNN network' % (tdnn_type))

        return nnet



    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if 'wo_norm' in kwargs:
            kwargs['use_norm'] = not kwargs['wo_norm']
            del kwargs['wo_norm']

        if 'norm_after' in kwargs:
            kwargs['norm_before'] = not kwargs['norm_after']
            del kwargs['norm_after']

        valid_args = ('tdnn_type', 'num_enc_blocks', 
                      'enc_hid_units', 'enc_expand_units', 'kernel_size',
                      'dilation', 'dilation_factor', 'in_norm', 'hid_act', 
                      'norm_layer', 'use_norm', 'norm_before', 'in_feats', 'dropout_rate')

        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        for arg in ('enc_hid_units', 'kernel_size', 'dilation'):
            if arg in args:
                val = args[arg] 
                if isinstance(val, list) and len(val)==1:
                    args[arg] = val[0]
        
        return args



    @staticmethod
    def add_argparse_args(parser, prefix=None):
        
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'

        parser.add_argument(p1+'tdnn-type', type=str.lower,
                            default='resetdnn',
                            choices=['tdnn','etdnn', 'resetdnn'],
                            help=('TDNN type: TDNN, ETDNN, ResETDNN'))

        parser.add_argument(p1+'num-enc-blocks',
                            default=9, type=int,
                            help=('number of encoder layer blocks'))

        parser.add_argument(p1+'enc-hid-units', nargs='+',
                            default=512, type=int,
                            help=('number of encoder layer blocks'))

        parser.add_argument(p1+'enc-expand-units',
                            default=None, type=int,
                            help=('dimension of last layer of ResETDNN'))

        parser.add_argument(p1+'kernel-size', nargs='+',
                            default=3, type=int,
                            help=('kernel sizes of encoder conv1d'))

        parser.add_argument(p1+'dilation', nargs='+',
                            default=1, type=int,
                            help=('dilations of encoder conv1d'))

        parser.add_argument(p1+'dilation-factor', 
                            default=1, type=int,
                            help=('dilation increment wrt previous conv1d layer'))

        try:
            parser.add_argument(p1+'hid-act', default='relu6', 
                                help='hidden activation')
        except:
            pass

        try:
            parser.add_argument(
                p1+'norm-layer', default=None, 
                choices=['batch-norm', 'group-norm', 'instance-norm', 'instance-norm-affine', 'layer-norm'],
                help='type of normalization layer')
        except:
            pass

        parser.add_argument(p1+'in-norm', default=False, action='store_true',
                            help='batch normalization at the input')

        try:
            parser.add_argument(p1+'wo-norm', default=False, action='store_true',
                                help='without batch normalization')
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

        try:
            parser.add_argument(p1+'in-feats', default=None, type=int,
                                help=('input feature dimension, '
                                      'if None it will try to infer from encoder network'))
        except:
            pass

