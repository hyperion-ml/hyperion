f"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn

from .xvector import XVector
from ..narchs import TransformerEncoderV1 as TE

class TransformerXVectorV1(XVector):

    def __init__(self, in_feats, num_classes,
                 enc_d_model=512,
                 num_enc_heads=4,
                 num_enc_blocks=6,
                 enc_att_type='scaled-dot-prod-v1',
                 enc_att_context = 25,
                 enc_ff_type='linear',
                 enc_d_ff=2048,
                 enc_ff_kernel_size=1,
                 in_layer_type='conv2d-sub',
                 enc_concat_after=False,
                 pool_net='mean+stddev', 
                 embed_dim=256,
                 num_embed_layers=1, 
                 hid_act={'name':'relu6', 'inplace':True}, 
                 loss_type='arc-softmax',
                 s=64, margin=0.3, margin_warmup_epochs=0,
                 dropout_rate=0.1,
                 pos_dropout_rate=0.1,
                 att_dropout_rate=0.0,
                 use_norm=True, 
                 norm_before=False,
                 in_norm=False, embed_layer=0, proj_feats=None):

        # if enc_expand_units is not None and isinstance(enc_hid_units, int):
        #     if tdnn_type != 'resetdnn' :
        #         enc_hid_units = (num_enc_blocks - 1)*[enc_hid_units] + [enc_expand_units]
        
        logging.info('making transformer-v1 encoder network')
        encoder_net = TE(
            in_feats,
            enc_d_model,
            num_enc_heads,
            num_enc_blocks,
            att_type = enc_att_type,
            att_context = enc_att_context,
            ff_type=enc_ff_type,
            d_ff=enc_d_ff,
            ff_kernel_size=enc_ff_kernel_size,
            ff_dropout_rate=dropout_rate,
            pos_dropout_rate=pos_dropout_rate,
            att_dropout_rate=att_dropout_rate,
            in_layer_type=in_layer_type,
            norm_before=norm_before,
            concat_after=enc_concat_after,
            in_time_dim=-1, out_time_dim=-1)
        
        super(TransformerXVectorV1, self).__init__(
            encoder_net, num_classes, pool_net=pool_net, 
            embed_dim=embed_dim, num_embed_layers=num_embed_layers, 
            hid_act=hid_act, loss_type=loss_type, 
            s=s, margin=margin, margin_warmup_epochs=margin_warmup_epochs,
            use_norm=use_norm, norm_before=norm_before, 
            dropout_rate=dropout_rate,
            embed_layer=embed_layer, 
            in_feats=None, proj_feats=proj_feats)

        
    @property
    def enc_d_model(self):
        return self.encoder_net.d_model

    @property
    def num_enc_heads(self):
        return self.encoder_net.num_heads

    @property
    def num_enc_blocks(self):
        return self.encoder_net.num_blocks

    @property
    def enc_att_type(self):
        return self.encoder_net.att_type

    @property
    def enc_att_context(self):
        return self.encoder_net.att_context

    @property
    def enc_ff_type(self):
        return self.encoder_net.ff_type 

    @property
    def enc_d_ff(self):
        return self.encoder_net.d_ff

    @property
    def enc_ff_kernel_size(self):
        return self.encoder_net.ff_kernel_size

    @property
    def pos_dropout_rate(self):
        return self.encoder_net.pos_dropout_rate


    @property
    def att_dropout_rate(self):
        return self.encoder_net.att_dropout_rate

    @property
    def in_layer_type(self):
        return self.encoder_net.in_layer_type

    @property
    def enc_concat_after(self):
        return self.encoder_net.concat_after

    @property
    def enc_ff_type(self):
        return self.encoder_net.ff_type 
    
    # @property
    # def in_norm(self):
    #     return self.encoder_net.in_norm


    def get_config(self):

        base_config = super(TransformerXVectorV1, self).get_config()
        del base_config['encoder_cfg']

        pool_cfg = self.pool_net.get_config()

        config = {'num_enc_blocks': self.num_enc_blocks, 
                  'in_feats': self.in_feats, 
                  'enc_d_model': self.enc_d_model, 
                  'num_enc_heads': self.num_enc_heads,
                  'enc_att_type': self.enc_att_type,
                  'enc_att_context': self.enc_att_context,
                  'enc_ff_type': self.enc_ff_type,
                  'd_enc_ff': self.enc_d_ff,
                  'enc_ff_kernel_size': self.enc_ff_kernel_size,
                  'pos_dropout_rate': self.pos_dropout_rate,
                  'att_dropout_rate': self.att_dropout_rate,
                  'in_layer_type': self.in_layer_type,
                  'enc_concat_after': self.enc_concat_after}
                  #'in_norm': self.in_norm }

        config.update(base_config)
        return config


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = TorchModel._load_cfg_state_dict(
            file_path, cfg, state_dict)

        model = cls(**cfg) 
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model


    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        base_args = XVector.filter_args(prefix, **kwargs)

        valid_args = ('num_enc_blocks',
                      'in_feats',
                      'enc_d_model',
                      'num_enc_heads',
                      'enc_att_type',
                      'enc_att_context',
                      'enc_ff_type',
                      'enc_d_ff',
                      'enc_ff_kernel_size',
                      'pos_dropout_rate',
                      'att_dropout_rate',
                      'in_layer_type',
                      'enc_concat_after')

        child_args = dict((k, kwargs[p+k])
                          for k in valid_args if p+k in kwargs)
        base_args.update(child_args)
        return base_args


    @staticmethod
    def add_argparse_args(parser, prefix=None):
        
        XVector.add_argparse_args(parser, prefix)
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'


        parser.add_argument(p1+'num-enc-blocks',
                            default=6, type=int,
                            help=('number of tranformer blocks'))

        parser.add_argument(p1+'enc-d-model', 
                            default=512, type=int,
                            help=('encoder layer sizes'))

        parser.add_argument(p1+'num-enc-heads',
                            default=4, type=int,
                            help=('number of heads in self-attention layers'))

        parser.add_argument(p1+'enc-att-type', 
                            default='scaled-dot-prod-v1', 
                            choices=['scaled-dot-prod-v1', 'local-scaled-dot-prod-v1'],
                            help=('type of self-attention'))

        parser.add_argument(p1+'enc-att-context', 
                            default=25, type=int,
                            help=('context size when using local attention'))

        parser.add_argument(p1+'enc-ff-type', 
                            default='linear', choices=['linear', 'conv1dx2', 'conv1dlinear'],
                            help=('type of feed forward layers in transformer block'))
        
        parser.add_argument(p1+'enc-d-ff',
                            default=2048, type=int,
                            help=('size middle layer in feed forward block')) 

        parser.add_argument(p1+'enc-ff-kernel-size',
                            default=3, type=int,
                            help=('kernel size in convolutional feed forward block')) 

        parser.add_argument(p1+'pos-dropout-rate', default=0.1, type=float,
                                help='positional encoder dropout')
        parser.add_argument(p1+'att-dropout-rate', default=0, type=float,
                                help='self-att dropout')
        
        parser.add_argument(p1+'in-layer-type', 
                            default='linear', choices=['linear', 'conv2d-sub'],
                            help=('type of input layer'))

        parser.add_argument(p1+'enc-concat-after', default=False, action='store_true',
                            help='concatenate attention input and output instead of adding')

        # parser.add_argument(p1+'in-norm', default=False, action='store_true',
        #                     help='batch normalization at the input')





