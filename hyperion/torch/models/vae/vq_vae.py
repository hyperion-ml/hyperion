"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.distributions as pdf
import torch.nn as nn

from ...layers import tensor2pdf as t2pdf
from ...layers import vq
from ...narchs import TorchNALoader
from ...hyper_torch_model import HyperTorchModel


class VQVAE(HyperTorchModel):
    """Vector-quantized variational autoencoder.

    Attributes:
      encoder_net: Encoder network.
      decoder_net: Decoder network.
      z_dim: Latent variable dimension.
      kldiv_weight: Weight of the KL divergence term in the ELBO.
      diversity_weight: Weight applied to the codebook perplexity term.
      vq_type: Vector quantizer type.
      vq_groups: Number of vector quantization groups.
      vq_clusters: Number of codewords in each VQ group.
      vq_commitment_cost: Commitment loss weight.
      vq_ema_gamma: Exponential moving average decay coefficient.
      vq_ema_eps: Laplace smoothing parameter.
      px_pdf: Data likelihood distribution type.
      flatten_spatial: If True, flatten spatial dimensions into one latent vector.
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
        diversity_weight: float = 0.1,
        vq_type: str = "multi-ema-k-means-vq",
        vq_groups: int = 1,
        vq_clusters: int = 64,
        vq_commitment_cost: float = 0.25,
        vq_ema_gamma: float = 0.99,
        vq_ema_eps: float = 1e-5,
        px_pdf: str = "normal-glob-diag-cov",
        flatten_spatial: bool = False,
        spatial_shape: Optional[tuple[int, ...]] = None,
        scale_invariant: bool = False,
        data_scale: Optional[float] = None,
    ) -> None:
        """Build a vector-quantized VAE.

        Args:
          encoder_net: Encoder network.
          decoder_net: Decoder network.
          z_dim: Latent dimensionality.
          kldiv_weight: Weight applied to the KL divergence term.
          diversity_weight: Weight applied to the codebook perplexity term.
          vq_type: Vector quantizer type.
          vq_groups: Number of vector quantizer groups.
          vq_clusters: Number of codewords per group.
          vq_commitment_cost: Commitment loss weight.
          vq_ema_gamma: EMA decay coefficient for EMA-based quantizers.
          vq_ema_eps: Laplace smoothing parameter for EMA-based quantizers.
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
          Tensor used by the VQ layer.
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

    def _make_vq_layer(self, in_feats: int, in_dim: int) -> None:
        """Instantiate the configured vector-quantizer layer.

        Args:
          in_feats: Input feature dimension to project from.
          in_dim: Input tensor rank.
        """

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
        x: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        return_x_mean: bool = False,
        return_x_sample: bool = False,
        return_z_sample: bool = False,
        return_px: bool = False,
        serialize_pdfs: bool = True,
    ) -> Dict[str, Any]:
        """Run the VQ-VAE forward pass.

        Args:
          x: Input tensor.
          x_target: Optional target tensor used to infer decoder output shape.
          return_x_mean: If True, include the likelihood mean in the output.
          return_x_sample: If True, include a sampled reconstruction in the output.
          return_z_sample: If True, include the quantized latent tensor in the output.
          return_px: If True, include the likelihood distribution in the output.
          serialize_pdfs: Reserved for API compatibility.

        Returns:
          Dictionary with loss and ELBO-related tensors.
        """

        if x_target is None:
            x_target = x

        xx = self._pre_enc(x)
        xx = self.encoder_net(xx)
        xx = self._post_enc(xx)

        vq_output = self.vq_layer(xx)
        z = vq_output.z_q
        codebook_loss = vq_output.codebook_loss
        commitment_loss = vq_output.commitment_loss
        vq_loss = codebook_loss + commitment_loss
        perplexity = vq_output.perplexity
        log_perplexity = torch.log(
            perplexity.clamp_min(torch.finfo(perplexity.dtype).tiny)
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
        # Normalize the regularizer by the number of elements in x for logging.
        kldiv_z = vq_loss / num_samples
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
            "perplexity": perplexity,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
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

        if return_px:
            r["px"] = px

        return r

    def compute_z(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the quantized latent tensor.

        Args:
          x: Input tensor.

        Returns:
          Quantized latent tensor.
        """
        x = self._pre_enc(x)
        xx = self.encoder_net(x)
        xx = self._post_enc(xx)

        vq_output = self.vq_layer(xx)
        return vq_output["z"]

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
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "VQVAE":
        """Load a VQ-VAE from configuration and state.

        Args:
          file_path: Optional checkpoint file path.
          cfg: Optional configuration dictionary.
          state_dict: Optional model state dictionary.

        Returns:
          Loaded VQ-VAE instance.
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
        """Filter keyword arguments relevant to the VQ-VAE.

        Args:
          **kwargs: Candidate keyword arguments.

        Returns:
          Filtered configuration dictionary.
        """
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
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Add VQ-VAE arguments to a parser.

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
