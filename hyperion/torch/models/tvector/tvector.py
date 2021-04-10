"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from argparse import Namespace

import torch
import torch.nn as nn

from ..layers import GlobalPool1dFactory as PF
from ..layer_blocks import TDNNBlock
from ..narchs import ClassifHead, ConformerEncoderV1
from ..torch_model import TorchModel
from ..helpers import TorchNALoader
from ..utils import eval_nnet_by_chunks

class TVector(TorchModel):
    """t-Vector base class
    """

    def __init__(self, encoder_net, num_classes, 
                 conformer_cfg=Namespace(
                     d_model=256, num_heads=4, num_blocks=6,
                     attype='scaled-dot-prod-v1', atcontext=25,
                     conv_repeats=1, conv_kernel_sizes=31, conv_strides=1,
                     ff_type='linear', d_ff=2048, ff_kernel_size=1,
                     dropourate=0.1, pos_dropourate=0.1, att_dropout_rate=0.0,
                     in_layer_type='conv2d-sub',
                     rel_pos_enc=True, causal_pos_enc=False, no_pos_enc=False,
                     hid_act='swish',
                     conv_norm_layer=None, se_r=None,
                     ff_macaron=True, red_lnorms=False, concat_after=False),
                 pool_net='mean+stddev', 
                 head_cfg=Namespace(
                     embed_dim=256,
                     num_embed_layers=1, 
                     head_hid_act={'name':'relu', 'inplace': True}, 
                     loss_type='arc-softmax',
                     s=64, margin=0.3, margin_warmup_epochs=0,
                     num_subcenters=2,
                     norm_layer=None,
                     use_norm=True, norm_before=True, 
                     dropout_rate=0,
                     embed_layer=0),
                 in_feats=None, proj_feats=None):

        super().__init__()

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

        if isinstance(conformer_cfg, Namespace):
            conformer_cfg = var(conformer_cfg)
        if isinstance(head_cfg, Namespace):
            head_cfg = var(head_cfg)

        self.conformer = ConformerEncoderV1(enc_feats, in_time_dim=1, out_time_dir=1, **conformer_cfg)
        self.proj_feats = self.conformer.d_model
        
        # create pooling network
        # infer output dimension of pooling which is input dim for classification head
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
            pool_feats, num_classes, **head_cfg)
        self.embed_layer = self.classif_net.embed_layer


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
    def num_subcenters(self):
        return self.classif_net.num_subcenters

    @property
    def loss_type(self):
        return self.classif_net.loss_type

    
    def _make_pool_net(self, pool_net, enc_feats=None):
        """ Makes the pooling block
        
        Args:
         pool_net: str or dict to pass to the pooling factory create function
         enc_feats: dimension of the features coming from the encoder

        Returns:
         GlobalPool1d object
        """
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
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
          epoch: epoch which is about to start
        """
        self.classif_net.update_margin(epoch)


    def _pre_enc(self, x):
        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        return x


    def _post_enc(self, x):
        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)
        
        return x

      
    def forward(self, x, y=None, use_amp=False):
        if use_amp:
            with torch.cuda.amp.autocast():
                return self._forward(x, y)

        return self._forward(x, y)

      
    def _forward(self, x, y=None):

        """Forward function

        Args:
          x: input features tensor with shape=(batch, in_feats, time)
          y: target classes torch.long tensor with shape=(batch,)
        
        Returns:
          class posteriors tensor with shape=(batch, num_classes)
        """
        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = self.encoder_net(x)

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)
    
        x = self.conformer_net(x)
        p = self.pool_net(x)
        y = self.classif_net(p, y)
        return y


    def forward_hid_feats(self, x, y=None, enc_layers=None, classif_layers=None, return_output=False):
        """forwards hidden representations in the x-vector network
        
        """

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



    def extract_embed(self, x, chunk_length=0, embed_layer=None, detach_chunks=False):
        if embed_layer is None:
            embed_layer = self.embed_layer

        x = self._pre_enc(x)
        # if self.encoder_net.in_dim() == 4 and x.dim() == 3:
        #     x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = eval_nnet_by_chunks(x, self.encoder_net, 
                                chunk_length, detach_chunks=detach_chunks)

        if x.device != self.device:
            x = x.to(self.device)

        x = self._post_enc(x)
        # if self.encoder_net.out_dim() == 4:
        #     x = x.view(x.size(0), -1, x.size(-1))

        # if self.proj is not None:
        #     x = self.proj(x)
        p = self.pool_net(x)
        y = self.classif_net.extract_embed(p, embed_layer)
        return y



    def extract_embed_slidwin(self, x, win_length, win_shift, snip_edges=False,
                              feat_frame_length=None, feat_frame_shift=None,
                              chunk_length=0, embed_layer=None, 
                              detach_chunks=False):

        if feat_frame_shift is not None:
            #assume win_length/shift are in secs, transform to frames
            # pass feat times from msecs to secs
            feat_frame_shift = feat_frame_shift / 1000
            feat_frame_length = feat_frame_length / 1000

            # get length and shift in number of feature frames
            win_shift = win_shift / feat_frame_shift # this can be a float
            win_length = (win_length - feat_frame_length + feat_frame_shift) / feat_frame_shift
            assert win_shift > 0.5, 'win-length should be longer than feat-frame-length'
            
        if embed_layer is None:
            embed_layer = self.embed_layer

        in_time = x.size(-1)
        # if self.encoder_net.in_dim() == 4 and x.dim() == 3:
        #     x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self._pre_enc(x)
        x = eval_nnet_by_chunks(
            x, self.encoder_net, 
            chunk_length, detach_chunks=detach_chunks)

        if x.device != self.device:
            x = x.to(self.device)

        x = self._post_enc(x)
        pin_time = x.size(-1)                # time dim before pooling
        downsample_factor = float(pin_time) / in_time
        p = self.pool_net.forward_slidwin(
            x, downsample_factor*win_length, downsample_factor*win_shift,
            snip_edges=snip_edges) 
        # (batch, pool_dim, time)

        p = p.transpose(1,2).contiguous().view(-1, p.size(1))
        y = self.classif_net.extract_embed(p, embed_layer).view(
            x.size(0), -1, self.embed_dim).transpose(1,2).contiguous()

        return y


    def compute_slidwin_timestamps(self, num_windows, win_length, win_shift, snip_edges=False, 
                                   feat_frame_length=25, feat_frame_shift=10, feat_snip_edges=False):

        P = self.compute_slidwin_left_padding(
            win_length, win_shift, snip_edges, 
            feat_frame_length, feat_frame_shift, feat_snip_edges)

        tstamps = torch.as_tensor([[i*win_shift, i*win_shift+win_length] for i in range(num_windows)]) - P
        tstamps[tstamps < 0] = 0
        return tstamps


    def compute_slidwin_left_padding(self, win_length, win_shift, snip_edges=False, 
                                     feat_frame_length=25, feat_frame_shift=10, feat_snip_edges=False):

        # pass feat times from msecs to secs
        feat_frame_shift = feat_frame_shift / 1000
        feat_frame_length = feat_frame_length / 1000

        # get length and shift in number of feature frames
        H = win_shift / feat_frame_shift
        L = (win_length - feat_frame_length + feat_frame_shift) / feat_frame_shift
        assert L > 0.5, 'win-length should be longer than feat-frame-length'
        
        # compute left padding in case of snip_edges is False
        if snip_edges:
            P1 = 0
        else:
            Q = (L - H) / 2 # left padding in frames introduced by x-vector sliding window
            P1 = Q * feat_frame_shift # left padding in secs introduced by x-vector sliding window


        if feat_snip_edges:
            # left padding introduced when computing acoustic feats
            P2 = 0
        else:
            P2 = (feat_frame_length - feat_frame_shift) / 2

        # total left padding
        return P1 + P2


    def get_config(self):

        enc_cfg = self.encoder_net.get_config()
        pool_cfg = PF.get_config(self.pool_net)
        conformer_cfg = self.conformer.get_config()
        head_cfg = self.classif_net.get_config()

        config = {'encoder_cfg': enc_cfg,
                  'pool_net': pool_cfg,
                  'num_classes': self.num_classes,
                  'conformer_cfg': conformer_cfg,
                  'head_cfg': head_cfg }
        
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(
            file_path, cfg, state_dict)

        # preproc_net = None
        # if 'preproc_cfg' in cfg:
        #     preproc_net = TorchNALoader.load(cfg=cfg['preproc_cfg'])
        #     del cfg['preproc_cfg']

        encoder_net = TorchNALoader.load_from_cfg(cfg=cfg['encoder_cfg'])
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



    def train_mode(self, mode='ft-embed-affine'):
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
            t_args = ConformerEncoderV1.filter_args(prefix='conformer',**kwargs)
            head_args = ClassifHead.filter_args(prefix='head',**kwargs)
        else:
            p = prefix + '_'
            t_args = ConformerEncoderV1.filter_args(prefix=prefix+'conformer',**kwargs)
            head_args = ClassifHead.filter_args(prefix='head',**kwargs)

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

        valid_args = ('num_classes', 
                      'in_feats', 'proj_feats')
        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        args['pool_net'] = pool_args
        args['conformer_cfg'] = t_args
        args['head_cfg'] = head_args
        return args


    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'
        
        ConformerEncoderV1.add_argparse_args(parser, prefix='t')
        ClassifHead.add_argparse_args(parser, prefix='head')
        
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
                            choices = ['softmax', 'arc-softmax', 'cos-softmax', 'subcenter-arc-softmax'],
                            help='loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax')
        
        parser.add_argument(p1+'s', default=64, type=float,
                            help='scale for arcface')
        
        parser.add_argument(p1+'margin', default=0.3, type=float,
                            help='margin for arcface, cosface,...')
        
        parser.add_argument(p1+'margin-warmup-epochs', default=10, type=float,
                            help='number of epoch until we set the final margin')

        parser.add_argument(p1+'num-subcenters', default=2, type=float,
                            help='number of subcenters in subcenter losses')
       
    



            
