"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn

from ..layers import GlobalPool1dFactory as PF
from ..layer_blocks import TDNNBlock
from ..narchs import ClassifHead
from ..torch_model import TorchModel
from ..helpers import TorchNALoader
from ..utils import eval_nnet_by_chunks

class XVector(TorchModel):

    def __init__(self, encoder_net, num_classes, pool_net='mean+stddev', 
                 embed_dim=256,
                 num_embed_layers=1, 
                 hid_act={'name':'relu', 'inplace': True}, 
                 loss_type='arc-softmax',
                 s=64, margin=0.3, margin_warmup_epochs=0,
                 use_norm=True, norm_before=True, 
                 dropout_rate=0,
                 embed_layer=0, 
                 in_feats=None, proj_feats=None):

        super(XVector, self).__init__()

        # encoder network
        self.encoder_net = encoder_net

        # infer input and output shapes of encoder network
        in_shape = self.encoder_net.in_shape()
        if len(in_shape) == 3:
            # encoder based on 1d conv or transformer
            in_feats = in_shape[1]
            out_shape = self.encoder_net.out_shape(in_shape)
            enc_feats = out_shape[1]
        elif len(in_shape) == 4:
            # encoder based in 2d convs
            assert in_feats is not None, 'in_feats dimension must be given to calculate pooling dimension'
            in_shape = list(in_shape)
            in_shape[2] = in_feats
            out_shape = self.encoder_net.out_shape(tuple(in_shape))
            enc_feats = out_shape[1]*out_shape[2]

        self.in_feats = in_feats

        logging.info('encoder input shape={}'.format(in_shape))
        logging.info('encoder output shape={}'.format(out_shape))

        # add projection network to link encoder and pooling layers if proj_feats is not None
        self.proj = None
        self.proj_feats = proj_feats
        if proj_feats is not None:
            logging.info('adding projection layer after encoder with in/out size %d -> %d' % (enc_feats, proj_feats)) 
            self.proj = TDNNBlock(enc_feats, proj_feats, kernel_size=1, 
                                  activation=None, use_norm=use_norm)

        
        # create pooling network

        #infer output dimension of pooling which is input dim for classification head
        if proj_feats is None:
            self.pool_net = self._make_pool_net(pool_net, enc_feats) 
            pool_feats = int(enc_feats * self.pool_net.size_multiplier)
        else:
            self.pool_net = self._make_pool_net(pool_net, proj_feats) 
            pool_feats = int(proj_feats * self.pool_net.size_multiplier)
        
        logging.info('infer pooling dimension %d' % (pool_feats))

        # create classification head
        logging.info('making classification head net')
        self.classif_net = ClassifHead(
            pool_feats, num_classes, embed_dim=embed_dim,
            num_embed_layers=num_embed_layers, 
            hid_act=hid_act,
            loss_type=loss_type,
            s=s, margin=margin, margin_warmup_epochs=margin_warmup_epochs,
            use_norm=use_norm, norm_before=norm_before, 
            dropout_rate=dropout_rate)

        self.hid_act = hid_act
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.dropout_rate = dropout_rate
        self.embed_layer = embed_layer

    @property
    def pool_feats(self):
        return self.classif_net.in_feats

    @property
    def num_classes(self):
        return self.classif_net.num_classes

    @property
    def embed_dim(self):
        return self.classif_net.embed_dim

    @property
    def num_embed_layers(self):
        return self.classif_net.num_embed_layers

    @property
    def s(self):
        return self.classif_net.s

    @property
    def margin(self):
        return self.classif_net.margin


    @property
    def margin_warmup_epochs(self):
        return self.classif_net.margin_warmup_epochs

    @property
    def loss_type(self):
        return self.classif_net.loss_type

    
    def _make_pool_net(self, pool_net, enc_feats=None):
        if isinstance(pool_net, str):
            pool_net = { 'pool_type': pool_net }

        if isinstance(pool_net, dict):
            if enc_feats is not None:
                pool_net['in_feats'] = enc_feats
            return PF.create(**pool_net)
        elif isinstance(pool_net, nn.Module):
            return pool_net
        else:
            raise Exception('Invalid pool_net argument')

    
    def update_loss_margin(self, epoch):
        self.classif_net.update_margin(epoch)


    def forward(self, x, y=None):

        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = self.encoder_net(x)

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)
            
        p = self.pool_net(x)
        y = self.classif_net(p, y)
        return y


    def forward_hid_feats(self, x, y=None, enc_layers=None, classif_layers=None, return_output=False):

        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        h_enc, x = self.encoder_net.forward_hid_feats(x, enc_layers, return_output=True)

        if not return_output and classif_layers is None:
            return h_enc
        
        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)
            
        p = self.pool_net(x)
        h_classif = self.classif_net.forward_hid_feats(p, y, classif_layers, return_output=return_output)
        if return_output:
            h_classif, y = h_classif
            return h_enc, h_classif, y

        return h_enc, h_classif


    # def forward(self, x, y=None):

    #     if self.encoder_net.in_dim() == 4 and x.dim() == 3:
    #         x = x.view(x.size(0), 1, x.size(1), x.size(2))

    #     x = self.encoder_net(x)

    #     if self.encoder_net.out_dim() == 4:
    #         x = x.view(x.size(0), -1, x.size(-1))

    #     if len(x)==24:
    #         mm = x.mean(dim=-1)
    #         mx = torch.max(x, dim=-1)[0]
    #         mn = torch.min(x, dim=-1)[0]
    #         na = torch.isnan(x).any(dim=-1)
    #         logging.info('resnet-mean={}'.format(str(mm[9])))
    #         logging.info('resnet-max={}'.format(str(mx[9])))
    #         logging.info('resnet-max2={}'.format(str(mx[9].max())))
    #         logging.info('resnet-min={}'.format(str(mn[9])))
    #         logging.info('resnet-min2={}'.format(str(mn[9].min())))
    #         logging.info('resnet-na={}'.format(str(na[9])))
    #         logging.info('resnet-na2={}'.format(str(na[9].any())))

    #     if self.proj is not None:
    #         x = self.proj(x)
            
    #     p = self.pool_net(x)
    #     if len(x)==24:
    #         logging.info('pool-max={}'.format(str(p[9].max())))
    #         logging.info('pool-min={}'.format(str(p[9].min())))
    #         logging.info('pool-na={}'.format(str(torch.isnan(p[9]).any())))

    #     y = self.classif_net(p, y)
    #     return y


    def extract_embed(self, x, chunk_length=0, embed_layer=None, device=None):
        if embed_layer is None:
            embed_layer = self.embed_layer

        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = eval_nnet_by_chunks(x, self.encoder_net, chunk_length, device=device)
        # if chunk_length == 0:
        #     if device is not None:
        #         x.to(device)
        #     x = self.encoder_net(x)
        # else:
        #     raise NotImplementedError()

        if device is not None:
            x = x.to(device)

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)

        p = self.pool_net(x)
        y = self.classif_net.extract_embed(p, embed_layer)
        return y


    def get_config(self):

        enc_cfg = self.encoder_net.get_config()
        pool_cfg = PF.get_config(self.pool_net)

        config = {'encoder_cfg': enc_cfg,
                  'pool_net': pool_cfg,
                  'num_classes': self.num_classes,
                  'embed_dim': self.embed_dim,
                  'num_embed_layers': self.num_embed_layers,
                  'hid_act': self.hid_act,
                  'loss_type': self.loss_type,
                  's': self.s,
                  'margin': self.margin,
                  'margin_warmup_epochs': self.margin_warmup_epochs,
                  'use_norm': self.use_norm,
                  'norm_before': self.norm_before,
                  'dropout_rate': self.dropout_rate,
                  'embed_layer': self.embed_layer,
                  'in_feats': self.in_feats,
                  'proj_feats': self.proj_feats }
        
        base_config = super(XVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(
            file_path, cfg, state_dict)

        # preproc_net = None
        # if 'preproc_cfg' in cfg:
        #     preproc_net = TorchNALoader.load(cfg=cfg['preproc_cfg'])
        #     del cfg['preproc_cfg']

        encoder_net = TorchNALoader.load(cfg=cfg['encoder_cfg'])

        for k in ('encoder_cfg'):
            del cfg[k]
        
        model = cls(encoder_net, **cfg) 
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model


    def rebuild_output_layer(self, num_classes=None, loss_type='arc-softmax', 
                             s=64, margin=0.3, margin_warmup_epochs=10):
        if (self.num_classes is not None and self.num_classes != num_classes) or (
                self.loss_type != loss_type):
            # if we change the number of classes or the loss-type
            # we need to reinitiate the last layer
            self.classif_net.rebuild_output_layer(
                num_classes, loss_type, s, margin, margin_warmup_epochs)
            return

        #otherwise we just change the values of s, margin and margin_warmup
        self.classif_net.set_margin(margin)
        self.classif_net.set_margin_warmup_epochs(margin_warmup_epochs)
        self.classif_net.set_s(s)


    def freeze_preembed_layers(self):
        self.encoder_net.freeze()
        if self.proj is not None:
            self.proj.freeze()

        for param in self.pool_net.parameters():
            param.requires_grad = False

        layer_list = [l for l in range(self.embed_layer)]
        self.classif_net.freeze_layers(layer_list)



    def train_mode(self, mode='ft_embed_affine'):
        if mode == 'ft-full' or mode == 'train':
            self.train()
            return 

        self.encoder_net.eval()
        if self.proj is not None:
            self.proj.eval()
        
        self.pool_net.eval()
        self.classif_net.train()
        layer_list = [l for l in range(self.embed_layer)]
        self.classif_net.put_layers_in_eval_mode(layer_list)

            

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        # get boolean args that are negated
        if 'pool_wo_bias' in kwargs:
            kwargs['pool_use_bias'] = not kwargs['pool_wo_bias']
            del kwargs['pool_wo_bias']

        if 'wo_norm' in kwargs:
            kwargs['use_norm'] = not kwargs['wo_norm']
            del kwargs['wo_norm']

        if 'norm_after' in kwargs:
            kwargs['norm_before'] = not kwargs['norm_after']
            del kwargs['norm_after']

        # get arguments for pooling
        pool_valid_args = (
            'pool_type', 'pool_num_comp', 'pool_use_bias', 
            'pool_dist_pow', 'pool_d_k', 'pool_d_v', 'pool_num_heads', 'pool_bin_attn')
        pool_args = dict((k, kwargs[p+k])
                         for k in pool_valid_args if p+k in kwargs)

        # remove pooling prefix from arg name
        for k in pool_valid_args[1:]:
            if k in pool_args:
                k2 = k.replace('pool_','')
                pool_args[k2] = pool_args[k]
                del pool_args[k]


        valid_args = ('num_classes', 'num_embed_layers', 'hid_act', 'loss_type',
                      's', 'margin', 'margin_warmup_epochs', 'use_norm', 'norm_before',
                      'in_feats', 'proj_feats', 'dropout_rate')
        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        args['pool_net'] = pool_args

        return args


    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'
        
        
        parser.add_argument(p1+'pool-type', type=str.lower,
                            default='mean+stddev',
                            choices=['avg','mean+stddev', 'mean+logvar', 
                                     'lde', 'scaled-dot-prod-att-v1'],
                            help=('Pooling methods: Avg, Mean+Std, Mean+logVar, LDE, '
                                  'scaled-dot-product-attention-v1'))
        
        parser.add_argument(p1+'pool-num-comp',
                            default=64, type=int,
                            help=('number of components for LDE pooling'))

        parser.add_argument(p1+'pool-dist-pow', 
                            default=2, type=int,
                            help=('Distace power for LDE pooling'))
        
        parser.add_argument(p1+'pool-wo-bias', 
                            default=False, action='store_true',
                            help=('Don\' use bias in LDE'))

        parser.add_argument(
            p1+'pool-num-heads', default=8, type=int,
            help=('number of attention heads'))

        parser.add_argument(
            p1+'pool-d-k', default=256, type=int,
            help=('key dimension for attention'))

        parser.add_argument(
            p1+'pool-d-v', default=256, type=int,
            help=('value dimension for attention'))

        parser.add_argument(
            p1+'pool-bin-attn', default=False, action='store_true',
            help=('Use binary attention, i.e. sigmoid instead of softmax'))

        # parser.add_argument(p1+'num-classes',
        #                     required=True, type=int,
        #                     help=('number of classes'))

        parser.add_argument(p1+'embed-dim',
                            default=256, type=int,
                            help=('x-vector dimension'))
        
        parser.add_argument(p1+'num-embed-layers',
                            default=1, type=int,
                            help=('number of layers in the classif head'))
        
        parser.add_argument(p1+'hid_act', default='relu6', 
                            help='hidden activation')

        parser.add_argument(p1+'loss-type', default='arc-softmax', 
                            choices = ['softmax', 'arc-softmax', 'cos-softmax'],
                            help='loss type: softmax, arc-softmax, cos-softmax')
        
        parser.add_argument(p1+'s', default=64, type=float,
                            help='scale for arcface')
        
        parser.add_argument(p1+'margin', default=0.3, type=float,
                            help='margin for arcface, cosface,...')
        
        parser.add_argument(p1+'margin-warmup-epochs', default=10, type=float,
                            help='number of epoch until we set the final margin')
        
        parser.add_argument(p1+'wo-norm', default=False, action='store_true',
                            help='without batch normalization')
        
        parser.add_argument(p1+'norm-after', default=False, action='store_true',
                            help='batch normalizaton after activation')
        
        parser.add_argument(p1+'dropout-rate', default=0, type=float,
                            help='dropout')
        
        parser.add_argument(p1+'in-feats', default=None, type=int,
                            help=('input feature dimension, '
                                  'if None it will try to infer from encoder network'))
        
        parser.add_argument(p1+'proj-feats', default=None, type=int,
                            help=('dimension of linear projection after encoder network, '
                                  'if None, there is not projection'))
        


    @staticmethod
    def filter_finetune_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        valid_args = ('loss_type', 's', 'margin', 'margin_warmup_epochs')
        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        return args


    @staticmethod
    def add_argparse_finetune_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'
        
        parser.add_argument(p1+'loss-type', default='arc-softmax', 
                            choices = ['softmax', 'arc-softmax', 'cos-softmax'],
                            help='loss type: softmax, arc-softmax, cos-softmax')
        
        parser.add_argument(p1+'s', default=64, type=float,
                            help='scale for arcface')
        
        parser.add_argument(p1+'margin', default=0.3, type=float,
                            help='margin for arcface, cosface,...')
        
        parser.add_argument(p1+'margin-warmup-epochs', default=10, type=float,
                            help='number of epoch until we set the final margin')
       
    



            
