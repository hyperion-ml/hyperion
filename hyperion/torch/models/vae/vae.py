"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.distributions as pdf
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...hyper_torch_model import HyperTorchModel
from ...layers import pdf_storage
from ...layers import tensor2pdf as t2pdf
from ...narchs import TorchNALoader


class VAE(HyperTorchModel):
    """Variational Autoencoder class
         From: https://arxiv.org/abs/1312.6114

    Attributes:
      encoder_net: Encoder network.
      decoder_net: Decoder network.
      z_dim: Latent variable dimension.
      kldiv_weight: Weight of the KL divergence term in the ELBO.
      qz_pdf: Approximate posterior distribution type.
      pz_pdf: Latent prior distribution type.
      px_pdf: Data likelihood distribution type.
      flatten_spatial: If True, use one latent vector per sample.
      spatial_shape: Input spatial shape required when ``flatten_spatial=True``.
      scale_invariant: Reserved for future use.
      data_scale: Reserved for future use.
    """

    def __init__(
        self,
        encoder_net: Any,
        decoder_net: Any,
        z_dim: int,
        kldiv_weight: float = 1,
        qz_pdf: str = "normal-glob-diag-cov",
        pz_pdf: str = "std-normal",
        px_pdf: str = "normal-glob-diag-cov",
        flatten_spatial: bool = False,
        spatial_shape: Optional[tuple[int, ...]] = None,
        scale_invariant: bool = False,
        data_scale: Optional[float] = None,
    ) -> None:
        """Build a variational autoencoder.

        Args:
          encoder_net: Encoder network.
          decoder_net: Decoder network.
          z_dim: Latent dimensionality.
          kldiv_weight: Weight applied to the KL divergence term.
          qz_pdf: Distribution used for the approximate posterior.
          pz_pdf: Distribution used for the prior.
          px_pdf: Distribution used for the likelihood.
          flatten_spatial: Whether to flatten spatial dimensions into one latent.
          spatial_shape: Input spatial shape when ``flatten_spatial=True``.
          scale_invariant: Reserved for future use.
          data_scale: Reserved for future use.
        """
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
    def pz(self) -> pdf.Distribution:
        """Latent prior distribution.

        Returns:
          Prior distribution over z.
        """
        return self._pz()

    def _compute_flatten_unflatten_shapes(self) -> None:
        """Infer flattening shapes for the spatial-latent mode."""
        # if we flatten the spatial dimension to have a single
        # latent representation for all time/spatial positions
        # we have to infer the spatial dimension at the encoder
        # output
        assert (
            self.spatial_shape is not None
        ), "you need to specify spatial shape at the input"

        enc_in_shape = None, self.in_channels, *self.spatial_shape
        self._enc_in_shape = enc_in_shape
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
        for d in self._dec_in_shape:
            dec_in_tot_feats *= d

        self._dec_in_tot_feats = dec_in_tot_feats

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten encoder output features.

        Args:
          x: Tensor to flatten.

        Returns:
          Flattened tensor.
        """
        return x.view(-1, self._enc_out_tot_feats)

    def _unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """Restore the decoder input shape.

        Args:
          x: Flattened tensor.

        Returns:
          Tensor reshaped for the decoder input.
        """
        return x.view(-1, *self._dec_in_shape)

    def _make_prior(self) -> None:
        """Instantiate the latent prior distribution."""

        if self.flatten_spatial:
            shape = (self.z_dim,)
        else:
            shape = self.z_dim, *(1,) * (self._enc_out_dim - 2)

        if self.pz_pdf == "std-normal":
            self._pz = pdf_storage.StdNormal(shape)
        else:
            raise ValueError("pz=%s not supported" % self.pz_pdf)

    def _make_t2pdf_layer(
        self, pdf_name: str, in_channels: int, channels: int, ndims: int
    ) -> nn.Module:
        """Build a tensor-to-distribution layer.

        Args:
          pdf_name: Distribution layer name.
          in_channels: Input feature dimension to project from.
          channels: Distribution parameter dimension.
          ndims: Input tensor rank.

        Returns:
          Instantiated tensor-to-distribution layer.
        """

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

    def _make_post_enc_layer(self) -> None:
        """Hook for building post-encoder layers."""

    def _make_pre_dec_layer(self) -> None:
        """Build the optional latent-to-decoder projection."""
        if self.flatten_spatial:
            self._pre_dec_linear = nn.Linear(self.z_dim, self._dec_in_tot_feats)

    def _make_post_dec_layer(self) -> None:
        """Hook for building post-decoder layers."""

    def _pre_enc(self, x: torch.Tensor) -> torch.Tensor:
        """Adjust input rank before encoding.

        Args:
          x: Input tensor.

        Returns:
          Tensor ready for the encoder.
        """
        if x.dim() == 3 and self._enc_in_dim == 4:
            return x.unsqueeze(1)

        return x

    def _post_enc(self, x: torch.Tensor) -> torch.Tensor:
        """Apply any post-encoder transformation.

        Args:
          x: Encoder output tensor.

        Returns:
          Tensor used by the posterior parameter layer.
        """
        if self.flatten_spatial:
            x = self._flatten(x)

        return x

    def _pre_dec(self, x: torch.Tensor) -> torch.Tensor:
        """Adjust latent rank before decoding.

        Args:
          x: Latent tensor.

        Returns:
          Tensor ready for the decoder.
        """
        if self.flatten_spatial:
            x = self._pre_dec_linear(x)
            x = self._unflatten(x)
            return x

        if self._enc_out_dim == 3 and self._dec_in_dim == 4:
            return x.unsqueeze(dim=1)

        if self._enc_out_dim == 4 and self._dec_in_dim == 3:
            return x.view(x.size(0), -1, x.size(-1))

        return x

    # def _post_px(self, px, x_shape):
    #     px_shape = px.batch_shape

    #     if len(px_shape) == 4 and len(x_shape) == 3:
    #         if px_shape[1] == 1:
    #             px = squeeze_pdf(px, dim=1)
    #         else:
    #             raise ValueError("P(x|z)-shape != x-shape")

    #     return px

    def forward(
        self,
        x: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        return_x_mean: bool = False,
        return_x_sample: bool = False,
        return_z_sample: bool = False,
        return_px: bool = False,
        return_qz: bool = False,
        serialize_pdfs: bool = True,
    ) -> Dict[str, Any]:
        """Run the VAE forward pass.

        Args:
          x: Input tensor.
          x_target: Optional target tensor used to infer decoder output shape.
          return_x_mean: If True, include the likelihood mean in the output.
          return_x_sample: If True, include a sampled reconstruction in the output.
          return_z_sample: If True, include the sampled latent vector in the output.
          return_px: If True, include the likelihood distribution in the output.
          return_qz: If True, include the posterior distribution in the output.
          serialize_pdfs: Reserved for API compatibility.

        Returns:
          Dictionary with ELBO-related tensors and optional extras.
        """

        if x_target is None:
            x_target = x

        x = self._pre_enc(x)
        xx = self.encoder_net(x)
        xx = self._post_enc(xx)
        qz = self.t2qz(xx, prior=self._pz())

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

        if return_px:
            r["px"] = px

        if return_qz:
            r["qz"] = qz

        return r

    def compute_qz(self, x: torch.Tensor) -> pdf.Distribution:
        """Compute the approximate posterior distribution.

        Args:
          x: Input tensor.

        Returns:
          Approximate posterior distribution q(z|x).
        """
        xx = self._pre_enc(x)
        xx = self.encoder_net(xx)
        xx = self._post_enc(xx)
        qz = self.t2qz(xx, prior=self.pz)
        return qz

    def compute_px_given_z(
        self, z: torch.Tensor, x_shape: Optional[torch.Size] = None
    ) -> pdf.Distribution:
        """Compute the data likelihood conditioned on a latent sample.

        Args:
          z: Latent tensor.
          x_shape: Optional target shape used by the decoder.

        Returns:
          Likelihood distribution p(x|z).
        """
        zz = self._pre_dec(z)

        zz = self.decoder_net(zz, target_shape=x_shape)

        if x_shape is None:
            x_shape = zz.shape
        squeeze_dim = None
        if len(x_shape) == 3 and zz.dim() == 4:
            squeeze_dim = 1
        px = self.t2px(zz, squeeze_dim=squeeze_dim)
        return px

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable model configuration.

        Returns:
          Configuration dictionary.
        """
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
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "VAE":
        """Load a VAE from configuration and state.

        Args:
          file_path: Optional checkpoint file path.
          cfg: Optional configuration dictionary.
          state_dict: Optional model state dictionary.

        Returns:
          Loaded VAE instance.
        """
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
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments relevant to the VAE.

        Args:
          **kwargs: Candidate keyword arguments.

        Returns:
          Filtered configuration dictionary.
        """
        valid_args = ("z_dim", "kldiv_weight", "qz_pdf", "px_pdf")
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Add VAE arguments to a parser.

        Args:
          parser: Argument parser to extend.
          prefix: Optional prefix for nested argument groups.
        """
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
