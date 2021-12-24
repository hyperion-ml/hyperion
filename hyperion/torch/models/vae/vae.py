"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn
import torch.distributions as pdf

from ...torch_model import TorchModel
from ...narchs import TorchNALoader
from ...layers import tensor2pdf as t2pdf
from ...layers import pdf_storage


class VAE(TorchModel):
    """Variational Autoencoder class
         From: https://arxiv.org/abs/1312.6114

    Attributes:
      encoder_net: NArch encoder network object
      decoder_net: NArch decoder network object
      z_dim: latent variable dimension
      kldiv_weight: weight KL divergene when computing ELBO
      qz_pdf: type of prob distribution of the approx. latent posterior
      pz_pdf: type of prob distribution of the latent prior
      px_pdf: type of prob distribution for the data likelihood
      flatten_spatial: if True all time/spatial dimensions are generated from a single latent vector,
                       if False, we have multiple latents depending on the data size.
      spatial_shape: shape of the data, only needed if flatten_spatial=True
      scale_invariant: for future use
      data_scale = for future use
    """

    def __init__(
        self,
        encoder_net,
        decoder_net,
        z_dim,
        kldiv_weight=1,
        qz_pdf="normal-glob-diag-cov",
        pz_pdf="std-normal",
        px_pdf="normal-glob-diag-cov",
        flatten_spatial=False,
        spatial_shape=None,
        scale_invariant=False,
        data_scale=None,
    ):
        super().__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.z_dim = z_dim
        self.qz_pdf = qz_pdf
        self.pz_pdf = pz_pdf
        self.px_pdf = px_pdf
        self.kldiv_weight = kldiv_weight
        self.flatten_spatial = flatten_spatial
        self.spatial_shape = spatial_shape
        self.scale_invariant = scale_invariant
        self.data_scale = data_scale

        # infer input feat dimension from encoder network
        in_shape = encoder_net.in_shape()
        # number of dimensions of input/output enc/dec tensors,
        # needed to connect the blocks
        self._enc_in_dim = len(in_shape)
        self._enc_out_dim = self.encoder_net.out_dim()
        self._dec_in_dim = self.decoder_net.in_dim()
        self._dec_out_dim = self.decoder_net.out_dim()

        # we assume conv nnets with channel in dimension 1
        self.in_channels = in_shape[1]
        self._enc_out_channels = self.encoder_net.out_shape()[1]
        self._dec_out_channels = self.decoder_net.out_shape()[1]

        if self.flatten_spatial:
            self._compute_flatten_unflatten_shapes()
            qz_in_channels = self._enc_out_tot_feats
            qz_in_dim = 2
        else:
            qz_in_channels = self._enc_out_channels
            qz_in_dim = self._enc_out_dim

        self._make_post_enc_layer()
        self._make_pre_dec_layer()
        self._make_post_dec_layer()

        self.t2qz = self._make_t2pdf_layer(
            qz_pdf, qz_in_channels, self.z_dim, qz_in_dim
        )
        self.t2px = self._make_t2pdf_layer(
            px_pdf, self._dec_out_channels, self.in_channels, self._dec_out_dim
        )

        self._make_prior()

    @property
    def pz(self):
        return self._pz()

    def _compute_flatten_unflatten_shapes(self):
        # if we flatten the spatial dimension to have a single
        # latent representation for all time/spatial positions
        # we have to infer the spatial dimension at the encoder
        # output
        assert (
            spatial_shape is not None
        ), "you need to specify spatial shape at the input"

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
        return x.view(-1, *self._dec_in_shape)

    def _make_prior(self):

        if self.flatten_spatial:
            shape = (self.z_dim,)
        else:
            shape = self.z_dim, *(1,) * (self._enc_out_dim - 2)

        if self.pz_pdf == "std-normal":
            self._pz = pdf_storage.StdNormal(shape)
        else:
            raise ValueError("pz=%s not supported" % self.pz_pdf)

    def _make_t2pdf_layer(self, pdf_name, in_channels, channels, ndims):

        pdf_dict = {
            "normal-i-cov": t2pdf.Tensor2NormalICov,
            "normal-glob-diag-cov": t2pdf.Tensor2NormalGlobDiagCov,
            "normal-diag-cov": t2pdf.Tensor2NormalDiagCov,
            "bay-normal-i-cov": t2pdf.Tensor2BayNormalICovGivenNormalPrior,
            "bay-normal-glob-diag-cov": t2pdf.Tensor2BayNormalGlobDiagCovGivenNormalPrior,
            "bay-normal-diag-cov": t2pdf.Tensor2BayNormalDiagCovGivenNormalPrior,
        }

        t2pdf_layer = pdf_dict[pdf_name](channels, in_feats=in_channels, in_dim=ndims)
        return t2pdf_layer

    def _make_post_enc_layer(self):
        pass

    def _make_pre_dec_layer(self):
        if self.flatten_spatial:
            self._pre_dec_linear = Linear(self.z_dim, self._dec_in_tot_dim)

    def _make_post_dec_layer(self):
        pass

    def _pre_enc(self, x):
        if x.dim() == 3 and self._enc_in_dim == 4:
            return x.unsqueeze(1)

        return x

    def _post_enc(self, x):
        if self.flatten_spatial:
            x = self._flatten(x)

        return x

    def _pre_dec(self, x):
        if self.flatten_spatial:
            x = self._prec_dec_linear(x)  # linear projection
            x = self._unflatten(x)
            return x

        if self._enc_out_dim == 3 and self._dec_in_dim == 4:
            return x.unsqueeze(dim=1)

        if self._enc_out_dim == 4 and self._dec_in_dim == 3:
            return x.view(x.size(0), -1, x.size(-1))

        return x

    def _post_px(self, px, x_shape):
        px_shape = px.batch_shape

        if len(px_shape) == 4 and len(x_shape) == 3:
            if px_shape[1] == 1:
                px = squeeze_pdf(px, dim=1)
            else:
                raise ValueError("P(x|z)-shape != x-shape")

        return px

    def forward(
        self,
        x,
        x_target=None,
        return_x_mean=False,
        return_x_sample=False,
        return_z_sample=False,
        return_px=False,
        return_qz=False,
        serialize_pdfs=True,
        use_amp=False,
    ):
        if use_amp:
            with torch.cuda.amp.autocast():
                return self._forward(
                    x,
                    x_target,
                    return_x_mean,
                    return_x_sample,
                    return_z_sample,
                    return_px,
                    return_qz,
                    serialize_pdfs,
                )

        return self._forward(
            x,
            x_target,
            return_x_mean,
            return_x_sample,
            return_z_sample,
            return_px,
            return_qz,
            serialize_pdfs,
        )

    def _forward(
        self,
        x,
        x_target=None,
        return_x_mean=False,
        return_x_sample=False,
        return_z_sample=False,
        return_px=False,
        return_qz=False,
        serialize_pdfs=True,
    ):

        if x_target is None:
            x_target = x

        x = self._pre_enc(x)
        xx = self.encoder_net(x)
        xx = self._post_enc(xx)
        qz = self.t2qz(xx, prior=self._pz())
        # print(qz)
        # print(self.pz)
        # print(qz.loc)
        # print(qz.scale)
        # print(self.pz.loc)
        # print(self.pz.scale)

        kldiv_qzpz = (
            pdf.kl.kl_divergence(qz, self._pz()).view(x.size(0), -1).sum(dim=-1)
        )
        z = qz.rsample()

        zz = self._pre_dec(z)
        zz = self.decoder_net(zz, target_shape=x_target.shape)

        squeeze_dim = None
        if x_target.dim() == 3 and zz.dim() == 4:
            squeeze_dim = 1
        px = self.t2px(zz, squeeze_dim=squeeze_dim)

        # we normalize the elbo by spatial/time samples and feature dimension
        log_px = px.log_prob(x_target).view(x.size(0), -1)

        num_samples = log_px.size(-1)
        log_px = log_px.mean(dim=-1)
        # kldiv must be normalized by number of elements in x, not in z!!
        kldiv_qzpz /= num_samples
        elbo = log_px - self.kldiv_weight * kldiv_qzpz

        # we build the return dict
        r = {"elbo": elbo, "log_px": log_px, "kldiv_z": kldiv_qzpz}

        if return_x_mean:
            r["x_mean"] = px.mean

        if return_x_sample:
            if px.has_rsample:
                x_sample = px.rsample()
            else:
                x_sample = px.sample()
            r["x_sample"] = x_sample

        if return_z_sample:
            r["z"] = z

        return r

    def compute_qz(self, x):
        xx = self._pre_enc(x)
        xx = self.encoder_net(xx)
        xx = self._post_enc(xx)
        qz = self.t2qz(xx, self.pz)
        return qz

    def compute_px_given_z(self, z, x_shape=None):
        zz = self._pre_dec(z)

        zz = self.decoder_net(zz, target_shape=x_shape)
        zz = self.pre_px(zz)

        squeeze_dim = None
        if x_target.dim() == 3 and zz.dim() == 4:
            squeeze_dim = 1
        px = self.t2px(zz, squeeze_dim=squeeze_dim)
        return px

    def get_config(self):
        enc_cfg = self.encoder_net.get_config()
        dec_cfg = self.decoder_net.get_config()
        config = {
            "encoder_cfg": enc_cfg,
            "decoder_cfg": dec_cfg,
            "z_dim": self.z_dim,
            "qz_pdf": self.qz_pdf,
            "pz_pdf": self.pz_pdf,
            "px_pdf": self.px_pdf,
            "kldiv_weight": self.kldiv_weight,
            "flatten_spatial": self.flatten_spatial,
            "spatial_shape": self.spatial_shape,
            "scale_invariant": self.scale_invariant,
            "data_scale": self.data_scale,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        encoder_net = TorchNALoader.load_from_cfg(cfg=cfg["encoder_cfg"])
        decoder_net = TorchNALoader.load_from_cfg(cfg=cfg["decoder_cfg"])
        for k in ("encoder_cfg", "decoder_cfg"):
            del cfg[k]

        model = cls(encoder_net, decoder_net, **cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("z_dim", "kldiv_weight", "qz_pdf", "px_pdf")
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--z-dim", type=int, required=True, help=("latent factor dimension")
        )

        parser.add_argument(
            "--kldiv-weight",
            default=1,
            type=float,
            help=("weight of the KL divergance in the ELBO"),
        )

        parser.add_argument(
            "--qz-pdf",
            default="normal-glob-diag-cov",
            choices=[
                "normal-i-cov",
                "normal-glob-diag-cov",
                "normal-diag-cov",
                "bay-normal-i-cov",
                "bay-normal-glob-diag-cov",
                "bay-normal-diag-cov",
            ],
            help=("pdf for approx posterior q(z)"),
        )

        parser.add_argument(
            "--px-pdf",
            default="normal-glob-diag-cov",
            choices=["normal-i-cov", "normal-glob-diag-cov", "normal-diag-cov"],
            help=("pdf for data likelihood p(x|z)"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='vae options')

    add_argparse_args = add_class_args
