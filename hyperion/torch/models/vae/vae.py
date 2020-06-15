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
from ...layers import pdf_storage
#import ...layers.tensor2pdf as t2pdf


class VAE(TorchModel):
    """Variational Autoencoder class
    """

    def __init__(self, encoder_net, decoder_net, z_dim, beta=1,
                 qz_pdf='normal-glob-diag-cov', pz_pdf='std-normal',
                 px_pdf='normal-glob-diag-cov',
                 flatten_spatial=False, spatial_shape=None):
        super().__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.z_dim = z_dim
        self.qz_pdf = qz_pdf
        self.pz_pdf = pz_pdf
        self.px_pdf = px_pdf
        self.beta = beta
        self.flatten_spatial = flatten_spatial
        self.spatial_shape = spatial_shape

        # infer input feat dimension from encoder network
        in_shape = encoder_net.in_shape()
        self._x_dims = len(in_shape) # number of dimension of input tensor, 
                                     # we will use this to know which kind of 
                                     # convolutions to use
        self._reduce_dims = tuple([i for i in range(1, self._x_dims)])
                                     
        # we assume conv nnets with channel in dimension 1
        self.in_feats = in_shape[1]

        if self.flatten_spatial:
            self._compute_flatten_unflatten_shapes()
            self.z2dec = Linear(self.z_dim, self._dec_in_tot_dim)

        self.t2qz = self._make_t2pdf_layer(qz_pdf, self.z_dim)
        self.t2px = self._make_t2pdf_layer(px_pdf, self.in_feats)

        self.pz = self._make_prior()

        self.pre_qz = self._make_pre_qz_layer()
        self.pre_px = self._make_pre_px_layer()

            
        
    def _compute_flatten_unflatten_shapes(self):
        # if we flatten the spatial dimension to have a single 
        # latent representation for all time/spatial positions
        # we have to infer the spatial dimension at the encoder 
        # output
        assert spatial_shape is not None, (
            'you need to specify spatial shape at the input')
        
        enc_in_shape = None, self.in_feats, *self.spatial_shape
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
        


    def _make_prior(self):

        if self.flatten_spatial:
            shape = (self.z_dim,)
        else:
            shape = self.z_dim, *(1,)*(self._x_dims - 2)

        if self.pz_pdf == 'std-normal':
            self._pz = pdf_storage.StdNormal(shape)
            # self._loc = nn.Parameter(torch.zeros(shape), requires_grad=False)
            # self._scale = nn.Parameter(torch.ones(shape), requires_grad=False)
            # pz = pdf.normal.Normal(self._loc, self._scale)
        else:
            raise ValueError('pz=%s not supported' % self.pz_pdf)

        return self._pz()



    def _make_t2pdf_layer(self, pdf_name, feat_dim):
        shape = feat_dim, *(1,)*(self._x_dims - 2)
        pdf_dict = { 
            'normal-glob-diag-cov': lambda : t2pdf.Tensor2NormalGlobDiagCov(shape),
            'normal-diag-cov': t2pdf.Tensor2NormalGlobDiagCov,
            'normal-i-cov': t2pdf.Tensor2NormalICov }

        t2pdf_layer = pdf_dict[pdf_name]()
        return t2pdf_layer



    def _make_conv1x1(self, in_feats, out_feats):
        ndims = self._x_dims
        if ndims == 2:
            return nn.Linear(in_feats, out_feats)
        elif ndims == 3:
            return nn.Conv1d(in_feats, out_feats, kernel_size=1)
        elif ndims == 4:
            return nn.Conv2d(in_feats, out_feats, kernel_size=1)
        elif ndims == 5:
            return nn.Conv3d(in_feats, out_feats, kernel_size=1)
        else:
            raise ValueError('ndim=%d is not supported' % ndims)
        


    def _make_pre_qz_layer(self):
        
        enc_feats = self.encoder_net.out_shape()[1]
        f = self.t2qz.tensor2pdfparam_factor
        if self.flatten_spatial:
            # we will need to pass channel dim to end dim and flatten
            pre_qz = Linear(self._enc_out_tot_feats, self.z_dim*f)
            return pre_qz

        return self._make_conv1x1(enc_feats, self.z_dim*f)

        
            
    def _make_pre_px_layer(self):
        dec_feats = self.decoder_net.out_shape()[1]
        f = self.t2px.tensor2pdfparam_factor
        return self._make_conv1x1(dec_feats, self.in_feats*f)
        
    
    def _match_sizes(self, y, target_shape):
        for i in range(2, self._x_dims):
            surplus = y.shape[i] - target_shape[i]
            if surplus > 0:
                y = torch.narrow(y, i, surplus//2, target_shape[i])

        return y.contiguous()

        
    def forward(self, x, x_target=None):
        
        if x_target is None:
            x_target = x
        
        xx = self.encoder_net(x)
        if self.flatten_spatial:
            xx = self._flatten(xx)

        xx = self.pre_qz(xx)
        qz = self.t2qz(xx, self._pz())
        # print(qz)
        # print(self.pz)
        # print(qz.loc)
        # print(qz.scale)
        # print(self.pz.loc)
        # print(self.pz.scale)

        kldiv_qzpz = pdf.kl.kl_divergence(qz, self._pz()).sum(
            dim=self._reduce_dims)
        z = qz.rsample()

        zz = z
        if self.flatten_spatial:
            zz = self._unflatten(self.z2dec(zz))

        zz = self.decoder_net(zz)
        zz = self.pre_px(zz)

        zz = self._match_sizes(zz, x.shape)
        px = self.t2px(zz)

        log_px = px.log_prob(x_target).sum(
            dim=self._reduce_dims)

        # print(z.shape, kldiv_qzpz.shape, x.shape, log_px.shape)
        #we normalize the elbo by the number of spatial/time samples
        num_samples = 1
        for d in x.shape[2:]:
            num_samples *= d

        log_px /= num_samples
        kldiv_qzpz /= num_samples
        elbo = log_px - self.beta*kldiv_qzpz

        return elbo, log_px, kldiv_qzpz, px, qz 
        


    def compute_qz(self, x):
        xx = self.encoder_net(x)
        if self.flatten_spatial:
            xx = self._flatten(xx)

        xx = self.pre_qz(xx)
        qz = self.t2qz(xx, self.pz)
        return qz


    def compute_px_given_z(self, z, x_shape=None):
        zz = z
        if self.flatten_spatial:
            zz = self._unflatten(self.z2dec(zz))

        zz = self.decoder_net(zz)
        zz = self.pre_px(zz)

        if x_shape is not None:
            zz = self._match_sizes(zz, x_shape)
        px = self.t2px(zz)
        return px


    def get_config(self):
        enc_cfg = self.encoder_net.get_config()
        dec_cfg = self.decoder_net.get_config()
        config = {'encoder_cfg': enc_cfg,
                  'decoder_cfg': dec_cfg,
                  'z_dim': self.z_dim,
                  'qz_pdf': self.qz_pdf,
                  'pz_pdf': self.pz_pdf,
                  'px_pdf': self.px_pdf,
                  'beta': self.beta,
                  'flatten_spatial': self.flatten_spatial,
                  'spatial_shape': self.spatial_shape }
        base_config = super(VAE, self).get_config()
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

        
