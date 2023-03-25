"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.distributions as pdf
import torch.nn as nn

from ...layers import tensor2pdf as t2pdf
from ...layers import vq
from ...narchs import TorchNALoader
from ...torch_model import TorchModel


class VQVAE(TorchModel):
    """Vector Quantized Variational Autoencoder class
          From: https://arxiv.org/abs/1711.00937

    Attributes:
      encoder_net: NArch encoder network object
      decoder_net: NArch decoder network object
      z_dim: latent variable dimension
      kldiv_weight: weight KL divergene when computing ELBO
      diversity_weight: weigth for log-perplexity of the codebook,
                        it inteds to maximize the number of codewords used.
      vq_type: type of vector quantizer
      vq_gropus: number of vector quantization groups.
      vq_clusters: number of codewords in each vq group
      vq_commitment_cost: weigth of the commitmenet loss
      vq_ema_gamma: exponential moving average decay coeff.
      vq_ema_eps: Laplace smoothing parameter
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
        diversity_weight=0.1,
        vq_type="multi-ema-k-means-vq",
        vq_groups=1,
        vq_clusters=64,
        vq_commitment_cost=0.25,
        vq_ema_gamma=0.99,
        vq_ema_eps=1e-5,
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
        self.px_pdf = px_pdf

        self.kldiv_weight = kldiv_weight
        self.diversity_weight = diversity_weight

        self.vq_type = vq_type
        self.vq_groups = vq_groups
        self.vq_clusters = vq_clusters
        self.vq_commitment_cost = vq_commitment_cost
        self.vq_ema_gamma = vq_ema_gamma
        self.vq_ema_eps = vq_ema_eps

        self.flatten_spatial = flatten_spatial
        self.spatial_shape = spatial_shape

        self.scale_invariant = scale_invariant
        self.data_scale = data_scale

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

        self._make_vq_layer(qz_in_channels, qz_in_dim)
        self.t2px = self._make_t2pdf_layer(
            px_pdf, self._dec_out_channels, self.in_channels, self._dec_out_dim
        )

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

    def _make_t2pdf_layer(self, pdf_name, in_channels, channels, ndims):

        pdf_dict = {
            "normal-i-cov": t2pdf.Tensor2NormalICov,
            "normal-glob-diag-cov": t2pdf.Tensor2NormalGlobDiagCov,
            "normal-diag-cov": t2pdf.Tensor2NormalDiagCov,
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

    def _make_vq_layer(self, in_feats, in_dim):

        if self.vq_type == "multi-k-means-vq":
            vq_layer = vq.MultiKMeansVectorQuantizer(
                self.vq_groups,
                self.vq_clusters,
                self.z_dim,
                self.vq_commitment_cost,
                in_feats=in_feats,
                in_dim=in_dim,
            )
        elif self.vq_type == "multi-ema-k-means-vq":
            vq_layer = vq.MultiEMAKMeansVectorQuantizer(
                self.vq_groups,
                self.vq_clusters,
                self.z_dim,
                self.vq_commitment_cost,
                self.vq_ema_gamma,
                self.vq_ema_eps,
                in_feats=in_feats,
                in_dim=in_dim,
            )
        elif self.vq_type == "k-means-vq":
            vq_layer = vq.KMeansVectorQuantizer(
                self.vq_clusters,
                self.z_dim,
                self.vq_commitment_cost,
                in_feats=in_feats,
                in_dim=in_dim,
            )
        elif self.vq_type == "ema-k-means-vq":
            vq_layer = vq.EMAKMeansVectorQuantizer(
                self.vq_clusters,
                self.z_dim,
                self.vq_commitment_cost,
                self.vq_ema_gamma,
                self.vq_ema_eps,
                in_feats=in_feats,
                in_dim=in_dim,
            )
        else:
            raise ValueError("vq_type=%s not supported" % (self.vq_type))

        self.vq_layer = vq_layer

    def forward(
        self,
        x,
        x_target=None,
        return_x_mean=False,
        return_x_sample=False,
        return_z_sample=False,
        return_px=False,
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
                    serialize_pdfs,
                )

        return self._forward(
            x,
            x_target,
            return_x_mean,
            return_x_sample,
            return_z_sample,
            return_px,
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
        serialize_pdfs=True,
    ):

        if x_target is None:
            x_target = x

        xx = self._pre_enc(x)
        xx = self.encoder_net(xx)
        xx = self._post_enc(xx)

        vq_output = self.vq_layer(xx)
        # extract the variables from the dict.
        z, vq_loss, kldiv_z, log_perplexity = (
            vq_output[i] for i in ["z_q", "loss", "kldiv_qrpr", "log_perplexity"]
        )
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
        kldiv_z /= num_samples
        elbo = log_px - self.kldiv_weight * kldiv_z

        loss = -elbo + vq_loss - self.diversity_weight * log_perplexity

        # we build the return dict
        r = {
            "loss": loss,
            "elbo": elbo,
            "log_px": log_px,
            "kldiv_z": kldiv_z,
            "vq_loss": vq_loss,
            "log_perplexity": log_perplexity,
        }

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

    def compute_z(self, x):
        x = self._pre_enc(x)
        xx = self.encoder_net(xx)
        xx = self._post_enc(xx)

        vq_output = self.vq_layer(xx)
        return vq_output["z"]

    def compute_px_given_z(self, z, x_shape=None):
        zz = self._pre_dec(z)
        zz = self.decoder_net(zz, target_shape=x_shape)
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
            "vq_type": self.vq_type,
            "vq_groups": self.vq_groups,
            "vq_clusters": self.vq_clusters,
            "vq_commitment_cost": self.vq_commitment_cost,
            "vq_ema_gamma": self.vq_ema_gamma,
            "vq_ema_eps": self.vq_ema_eps,
            "px_pdf": self.px_pdf,
            "kldiv_weight": self.kldiv_weight,
            "diversity_weight": self.diversity_weight,
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
        valid_args = (
            "z_dim",
            "kldiv_weight",
            "diversity_weight",
            "vq_type",
            "vq_groups",
            "vq_clusters",
            "vq_commitment_cost",
            "vq_ema_gamma",
            "vq_ema_eps",
            "px_pdf",
        )

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
            "--diversity-weight",
            default=0.1,
            type=float,
            help=("weight of the log-perplexity in the loss"),
        )

        parser.add_argument(
            "--vq-type",
            default="ema-k-means-vq",
            choices=[
                "k-means-vq",
                "multi-k-means-vq",
                "ema-k-means-vq",
                "multi-ema-k-means-vq",
            ],
            help=("type of vector quantization layer"),
        )

        parser.add_argument(
            "--vq-groups",
            default=1,
            type=int,
            help=("number of groups in mulit-vq layers"),
        )

        parser.add_argument(
            "--vq-clusters", default=64, type=int, help=("size of the codebooks")
        )

        parser.add_argument(
            "--vq-commitment-cost",
            default=0.25,
            type=float,
            help=("commitment loss weight (beta in VQ-VAE paper)"),
        )

        parser.add_argument(
            "--vq-ema-gamma",
            default=0.99,
            type=float,
            help=(
                "decay parameter for exponential moving "
                "average calculation of the embeddings"
            ),
        )

        parser.add_argument(
            "--vq-ema-eps",
            default=1e-5,
            type=float,
            help=(
                "pseudo-count value for Laplace smoothing "
                "of cluster counts for exponential moving "
                "avarage calculation of the embeddings"
            ),
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
