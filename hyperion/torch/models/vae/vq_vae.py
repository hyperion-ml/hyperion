"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn
import torch.distributions as pdf

from ...torch_model import TorchModel
from ...helpers import TorchNALoader
from ...layers import tensor2pdf as t2pdf
from ...layers import vq 
from ...layers import pdf_storage
from ...utils.distributions import squeeze_pdf

class VQVAE(TorchModel):
    """Variational Autoencoder class
    """

    def __init__(self, encoder_net, decoder_net, z_dim, kldiv_weight=1,
                 vq_type='multi-ema-k-means-vq', vq_groups=1, vq_clusters=64, 
                 vq_commitment_cost=0.25, vq_ema_gamma=0.99, vq_ema_eps=1e-5,
                 px_pdf='normal-glob-diag-cov',
                 flatten_spatial=False, spatial_shape=None):

        super().__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.z_dim = z_dim
        self.px_pdf = px_pdf
        self.kldiv_weight = kldiv_weight
        self.vq_type = vq_type
        self.vq_groups = vq_groups
        self.vq_clusters = vq_clusters
        self.vq_commitment_cost = vq_commitment_cost
        self.vq_ema_gamma = vq_ema_gamma
        self.vq_ema_eps = vq_ema_eps
        
        self.flatten_spatial = flatten_spatial
        self.spatial_shape = spatial_shape

        # infer input feat dimension from encoder network
        in_shape = encoder_net.in_shape()
        # number of dimension of input/output enc/dec tensors, 
        # needed to connect the blocks
        self._enc_in_dim = len(in_shape) 
        self._enc_out_dim = self.encoder_net.out_dim()
        self._dec_in_dim = self.decoder_net.in_dim()
        self._dec_out_dim = self.decoder_net.out_dim()

        # we assume conv nnets with channel in dimension 1
        self.in_channels = in_shape[1]

        if self.flatten_spatial:
            self._compute_flatten_unflatten_shapes()
            self.z2dec = Linear(self.z_dim, self._dec_in_tot_dim)

        self.vq_layer = self._make_vq_layer()
        self.t2px = self._make_t2pdf_layer(px_pdf, self.in_channels, self._dec_out_dim)

        self.pre_vq = self._make_pre_vq_layer()
        self.pre_px = self._make_pre_px_layer()

            
        
    def _compute_flatten_unflatten_shapes(self):
        # if we flatten the spatial dimension to have a single 
        # latent representation for all time/spatial positions
        # we have to infer the spatial dimension at the encoder 
        # output
        assert spatial_shape is not None, (
            'you need to specify spatial shape at the input')
        
        enc_in_shape = None, self.in_channels, *self.spatial_shape
        enc_out_shape = self.encoder_net.out_shape(enc_in_shape)
        self._enc_out_shape = enc_out_shape[1:]

        # this is the total number of flattened features at the encoder output
        enc_out_tot_feats = 1
        for d in self._enc_out_shape:
            enc_out_tot_feats *= d

        self._enc_out_tot_feats = enc_out_tot_feats

        # now we infer the shape at the decoder input
        dec_in_shape = self.decoder_net.in_shape()
        # we keep the spatial dims at the encoder output
        self._dec_in_shape = dec_in_shape[1], *enc_out_shape[2:]
        # this is the total number of flattened features at the decoder input
        dec_in_tot_feats = 1
        for d in self._enc_in_shape:
            dec_in_tot_feats *= d
        
        self._dec_in_tot_feats = dec_in_tot_feats



    def _flatten(self, x):
        return x.view(-1, self._enc_out_tot_feats)



    def _unflatten(sef, x):
        x = self.z2dec(x) #linear projection
        return x.view(-1, *self._dec_in_shape)
        


    def _make_t2pdf_layer(self, pdf_name, channels, ndims):
        shape = channels, *(1,)*(ndims - 2)
        pdf_dict = { 
            'normal-glob-diag-cov': lambda : t2pdf.Tensor2NormalGlobDiagCov(shape),
            'normal-diag-cov': t2pdf.Tensor2NormalGlobDiagCov,
            'normal-i-cov': t2pdf.Tensor2NormalICov }

        t2pdf_layer = pdf_dict[pdf_name]()
        return t2pdf_layer



    def _make_conv1x1(self, in_channels, out_channels, ndims):
        if ndims == 2:
            return nn.Linear(in_channels, out_channels)
        elif ndims == 3:
            return nn.Conv1d(in_channels, out_channels, kernel_size=1)
        elif ndims == 4:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif ndims == 5:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            raise ValueError('ndim=%d is not supported' % ndims)
        


    def _make_pre_vq_layer(self):
        
        enc_channels = self.encoder_net.out_shape()[1]
        if self.flatten_spatial:
            # we will need to pass channel dim to end dim and flatten
            pre_vq = Linear(self._enc_out_tot_feats, self.z_dim)
            return pre_vq

        return self._make_conv1x1(enc_channels, self.z_dim, self._enc_out_dim)

        
            
    def _make_pre_px_layer(self):
        dec_channels = self.decoder_net.out_shape()[1]
        f = self.t2px.tensor2pdfparam_factor
        return self._make_conv1x1(dec_channels, self.in_channels*f, self._dec_out_dim)
        

    
    def _match_sizes(self, y, target_shape):
        y_dim = len(y.shape)
        d = y_dim - len(target_shape)
        for i in range(2, y_dim):
            surplus = y.shape[i] - target_shape[i-d]
            if surplus > 0:
                y = torch.narrow(y, i, surplus//2, target_shape[i])

        return y.contiguous()



    def _pre_enc(self, x):
        if x.dim() == 3 and self._enc_in_dim == 4:
            return x.unsqueeze(1)

        return x
        


    def _post_px(self, px, x_shape):
        px_shape = px.batch_shape
        
        if len(px_shape) == 4 and len(x_shape)==3:
            if px_shape[1]==1:
                px = squeeze_pdf(px, dim=1)
            else:
                raise ValueError('P(x|z)-shape != x-shape')
            
        return px



    def _make_vq_layer(self):
        if self.vq_type == 'multi-k-means-vq':
            return vq.MultiKMeansVectorQuantizer(
                self.vq_groups, self.vq_clusters, self.z_dim, 
                self.vq_commitment_cost)
        elif self.vq_type == 'multi-ema-k-means-vq':
            return vq.MultiEMAKMeansVectorQuantizer(
                self.vq_groups, self.vq_clusters, self.z_dim, 
                self.vq_commitment_cost, self.vq_ema_gamma, self.vq_ema_eps)
        elif self.vq_type == 'k-means-vq':
            return vq.KMeansVectorQuantizer(
                self.vq_clusters, self.z_dim, 
                self.vq_commitment_cost)
        elif self.vq_type == 'ema-k-means-vq':
            return vq.EMAKMeansVectorQuantizer(
                self.vq_clusters, self.z_dim, 
                self.vq_commitment_cost, self.vq_ema_gamma, self.vq_ema_eps)
        else:
            raise ValueError('vq_type=%s not supported' % (self.vq_type))
            

        
    def forward(self, x, x_target=None, 
                return_x_mean=False,
                return_x_sample=False, return_z_sample=False,
                return_px=False, return_qz=False, serialize_pdfs=True):
        
        if x_target is None:
            x_target = x
        
        x = self._pre_enc(x)
        xx = self.encoder_net(x)
        if self.flatten_spatial:
            xx = self._flatten(xx)

        xx = self.pre_vq(xx)

        z, vq_loss, kldiv_qzpz, perplexity = self.vq_layer(xx)

        zz = z
        if self.flatten_spatial:
            zz = self._unflatten(zz)

        zz = self.decoder_net(zz)
        zz = self.pre_px(zz)
        zz = self._match_sizes(zz, x_target.shape)
        px = self.t2px(zz)
        px = self._post_px(px, x_target.shape)

        # we normalize the elbo by spatial/time samples and feature dimension
        log_px = px.log_prob(x_target).view(
            x.size(0), -1)

        num_samples = log_px.size(-1)
        log_px = log_px.mean(dim=-1)
        # kldiv must be normalized by number of elements in x, not in z!!
        kldiv_qzpz /= num_samples 
        elbo = log_px - self.kldiv_weight*kldiv_qzpz

        loss = - elbo + vq_loss

        # we build the return tuple
        r = [loss, elbo, log_px, kldiv_qzpz, vq_loss, perplexity]
        if return_x_mean:
            r.append(px.mean)

        if return_x_sample:
            if px.has_rsample:
                x_tilde = px.rsample()
            else:
                x_tilde = px.sample()
            
            r.append(x_tilde)

        if return_z_sample:
            r.append(z)

        return tuple(r)
        


    def compute_z(self, x):
        x = self._pre_enc(x)
        xx = self.encoder_net(x)
        if self.flatten_spatial:
            xx = self._flatten(xx)

        xx = self.pre_vq(xx)

        z, vq_loss, kldiv_qzpz, perplexity = self.vq_layer(xx)
        return z


    def compute_px_given_z(self, z, x_shape=None):
        zz = z
        if self.flatten_spatial:
            zz = self._unflatten(self.z2dec(zz))

        zz = self.decoder_net(zz)
        zz = self.pre_px(zz)

        if x_shape is not None:
            zz = self._match_sizes(zz, x_shape)
        px = self.t2px(zz)
        if x_shape is not None:
            px = self._post_px(px, x_shape)
        return px


    def get_config(self):
        enc_cfg = self.encoder_net.get_config()
        dec_cfg = self.decoder_net.get_config()
        config = {'encoder_cfg': enc_cfg,
                  'decoder_cfg': dec_cfg,
                  'z_dim': self.z_dim,
                  'vq_type': self.vq_type,
                  'vq_groups': self.vq_groups,
                  'vq_clusters': self.vq_clusters,
                  'vq_commitment_cost': self.vq_commitment_cost,
                  'vq_ema_gamma': self.vq_ema_gamma,
                  'vq_ema_eps': self.vq_ema_eps,
                  'px_pdf': self.px_pdf,
                  'kldiv_weight': self.kldiv_weight,
                  'flatten_spatial': self.flatten_spatial,
                  'spatial_shape': self.spatial_shape }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(
            file_path, cfg, state_dict)

        encoder_net = TorchNALoader.load_from_cfg(cfg=cfg['encoder_cfg'])
        decoder_net = TorchNALoader.load_from_cfg(cfg=cfg['decoder_cfg'])
        for k in ('encoder_cfg', 'decoder_cfg'):
            del cfg[k]
        
        model = cls(encoder_net, decoder_net, **cfg) 
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

        


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        valid_args = ('z_dim', 'kldiv_weight', 'vq_type', 'vq_groups', 'vq_clusters',
                      'vq_commitment_cost', 'vq_ema_gamma', 'vq_ema_eps')

        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        return args



    @staticmethod
    def add_argparse_args(parser, prefix=None):
        
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'

        parser.add_argument(
                p1+'z-dim', type=int, required=True,
                help=('latent factor dimension'))

        parser.add_argument(p1+'kldiv-weight', default=1, type=float,
                            help=('weight of the KL divergance in the ELBO'))

        parser.add_argument(
            p1+'vq-type', default='ema-k-means-vq', 
            choices = ['k-means-vq', 'multi-k-means-vq', 'ema-k-means-vq', 'multi-ema-k-means-vq'],
            help=('type of vector quantization layer'))

        parser.add_argument(
            p1+'vq-groups', default=1, type=int,
            help=('number of groups in mulit-vq layers'))

        parser.add_argument(
            p1+'vq-clusters', default=64, type=int,
            help=('size of the codebooks'))

        parser.add_argument(p1+'vq-commitment-cost', default=0.25, type=float,
                            help=('commitment loss weight (beta in VQ-VAE paper)'))

        parser.add_argument(p1+'vq-ema-gamma', default=0.99, type=float,
                            help=('decay parameter for exponential moving '
                                  'average calculation of the embeddings'))

        parser.add_argument(p1+'vq-ema-eps', default=1e-5, type=float,
                            help=('pseudo-count value for Laplace smoothing '
                                  'of cluster counts for exponential moving '
                                  'avarage calculation of the embeddings'))

