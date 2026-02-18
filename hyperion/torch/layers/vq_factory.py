"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .vq import (
    AdaptiveRateDistortionEMVectorQuantizer,
    BinarySplittingGMMVectorQuantizer,
    EMAGumbelVectorQuantizer,
    EMANNVectorQuantizer,
    GumbelVectorQuantizer,
    NNVectorQuantizer,
    VQDistanceType,
)

vq_dict = {
    "nn_vq": NNVectorQuantizer,
    "ema_nn_vq": EMANNVectorQuantizer,
    "gumbel_vq": GumbelVectorQuantizer,
    "ema_gumbel_vq": EMAGumbelVectorQuantizer,
    "binary_splitting_gmm_vq": BinarySplittingGMMVectorQuantizer,
    "adaptive_rd_em_vq": AdaptiveRateDistortionEMVectorQuantizer,
}


class VectorQuantizerFactory:
    """Factory class for vector quantizers."""

    @staticmethod
    def create(
        vq_type: str,
        in_feats: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        distance_metric: str | VQDistanceType = VQDistanceType.L2,
        use_weight_norm: bool = False,
        channels_last: bool = False,
        **kwargs,
    ) -> nn.Module:
        """Creates a vector quantizer based on the specified type and parameters.

        Args:
            vq_type (str): Type of vector quantizer to create. Options are:
                           'nn_vq', 'ema_nn_vq', 'gumbel_vq', 'ema_gumbel_vq'.
            in_feats (int): Input feature dimension.
            codebook_size (int): Number of embeddings (K) in the codebook.
            codebook_dim (Optional[int]): Dimension of each embedding (D) in the codebook.
                                          If None, defaults to `in_feats`.
            distance_metric (str): Distance metric for nearest neighbor search.
                                   Options are defined in `VQDistanceType`.
            use_weight_norm (bool): Whether to apply weight normalization to the codebook embeddings.
            channels_last (bool): Whether to use channels_last memory format for inputs.
            **kwargs: Additional keyword arguments specific to the vector quantizer type.

        Returns:
            nn.Module: An instance of the specified vector quantizer.
        """
        if vq_type not in vq_dict:
            raise ValueError(
                f"Unsupported vq_type '{vq_type}'. Supported types are: {list(vq_dict.keys())}"
            )

        # Normalize distance metric to enum
        if isinstance(distance_metric, str):
            distance_metric = VQDistanceType(distance_metric)

        vq_class = vq_dict[vq_type]
        vq_params = {
            "in_feats": in_feats,
            "codebook_size": codebook_size,
            "codebook_dim": codebook_dim,
            "distance_metric": distance_metric,
            "use_weight_norm": use_weight_norm,
            "channels_last": channels_last,
        }
        if vq_class in (
            BinarySplittingGMMVectorQuantizer,
            AdaptiveRateDistortionEMVectorQuantizer,
        ):
            # Remove params not used by diagonal-GMM quantizers.
            vq_params.pop("distance_metric")

        vq_params.update(kwargs)
        vq_params = vq_class.filter_args(**vq_params)
        return vq_class(**vq_params)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Optional[set] = None
    ):

        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "vq_type" not in skip:
            parser.add_argument(
                "--vq-type",
                choices=list(vq_dict.keys()),
                default="nn_vq",
                help="Type of vector quantizer to use",
            )

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats",
                type=int,
                required=True,
                help="Input feature dimension to the vector quantizer",
            )

        if "codebook_size" not in skip:
            parser.add_argument(
                "--codebook-size",
                type=int,
                required=True,
                help="Number of embeddings (K) in the codebook",
            )

        if "codebook_dim" not in skip:
            parser.add_argument(
                "--codebook-dim",
                type=int,
                default=None,
                help="Dimension of each embedding (D) in the codebook. "
                "If not provided, defaults to `in_feats`",
            )

        if "distance_metric" not in skip:
            parser.add_argument(
                "--distance-metric",
                choices=VQDistanceType.choices(),
                default=VQDistanceType.L2.value,
                help="Distance metric to use for nearest neighbor search",
            )

        if "use_weight_norm" not in skip:
            parser.add_argument(
                "--use-weight-norm",
                action=ActionYesNo,
                default=False,
                help="Whether to apply weight normalization to the codebook embeddings",
            )

        if "channels_last" not in skip:
            parser.add_argument(
                "--channels-last",
                action=ActionYesNo,
                default=False,
                help="Whether to use channels_last memory format for inputs",
            )

        if "decay" not in skip:
            parser.add_argument(
                "--decay",
                type=float,
                default=0.99,
                help="Decay factor for exponential moving average",
            )

        if "eps" not in skip:
            parser.add_argument(
                "--eps",
                type=float,
                default=1e-5,
                help="Epsilon value for numerical stability",
            )

        if "reset_unused" not in skip:
            parser.add_argument(
                "--reset-unused",
                action=ActionYesNo,
                default=False,
                help="Whether to reset codebook entries whose EMA usage drops below the threshold",
            )

        if "ema_usage_threshold" not in skip:
            parser.add_argument(
                "--ema-usage-threshold",
                type=float,
                default=1.0,
                help="EMA usage level below which a codebook entry is considered unused and reset",
            )

        if "init_cluster_size" not in skip:
            parser.add_argument(
                "--init-cluster-size",
                type=float,
                default=1.0,
                help="Initial EMA cluster size for each codebook entry",
            )

        if "split_start_steps" not in skip:
            parser.add_argument(
                "--split-start-steps",
                type=int,
                default=0,
                help=(
                    "Global step offset before the first split "
                    "(first split at split_start_steps + split_steps)"
                ),
            )

        if "split_steps" not in skip:
            parser.add_argument(
                "--split-steps",
                type=int,
                default=1000,
                help="Number of training steps between split events",
            )

        if "split_std_scale" not in skip:
            parser.add_argument(
                "--split-std-scale",
                type=float,
                default=0.75,
                help=(
                    "Scale factor for the standard deviation used to offset split "
                    "means along the maximum-variance dimension (must be in (0, 1))"
                ),
            )

        if "max_weight_ratio" not in skip:
            parser.add_argument(
                "--max-weight-ratio",
                type=float,
                default=2.0,
                help="Maximum weight ratio before resetting unused codes",
            )

        if "reset_cooldown_steps" not in skip:
            parser.add_argument(
                "--reset-cooldown-steps",
                type=int,
                default=1000,
                help="Minimum number of steps between resets for a codebook entry",
            )

        if "var_floor" not in skip:
            parser.add_argument(
                "--var-floor",
                type=float,
                default=1e-5,
                help="Minimum variance for precision updates",
            )

        if "usage_target_ppl_ratio" not in skip:
            parser.add_argument(
                "--usage-target-ppl-ratio",
                type=float,
                default=0.5,
                help="Target perplexity ratio (target_ppl=ratio*codebook_size)",
            )

        if "usage_lambda_init" not in skip:
            parser.add_argument(
                "--usage-lambda-init",
                type=float,
                default=0.0,
                help="Initial usage regularization factor",
            )

        if "usage_lambda_lr" not in skip:
            parser.add_argument(
                "--usage-lambda-lr",
                type=float,
                default=1e-3,
                help="Step size for adaptive usage regularization factor",
            )

        if "usage_lambda_max" not in skip:
            parser.add_argument(
                "--usage-lambda-max",
                type=float,
                default=0.95,
                help="Maximum value for adaptive usage regularization factor",
            )

        if "dirichlet_alpha" not in skip:
            parser.add_argument(
                "--dirichlet-alpha",
                type=float,
                default=0.0,
                help="Dirichlet pseudocount for mixture-weight smoothing",
            )

        if "temp_init" not in skip:
            parser.add_argument(
                "--temp-init",
                type=float,
                default=1.0,
                help="Initial temperature for Gumbel softmax",
            )

        if "temp_min" not in skip:
            parser.add_argument(
                "--temp-min",
                type=float,
                default=0.5,
                help="Minimum temperature for Gumbel softmax",
            )

        if "temp_anneal_rate" not in skip:
            parser.add_argument(
                "--temp-anneal-rate",
                type=float,
                default=1e-5,
                help="Annealing rate for temperature",
            )

        if "temp_anneal_steps" not in skip:
            parser.add_argument(
                "--temp-anneal-steps",
                type=int,
                default=None,
                help="Number of steps to reach temp_min (overrides temp_anneal_rate)",
            )

        if "soft_sampling_steps" not in skip:
            parser.add_argument(
                "--soft-sampling-steps",
                type=int,
                default=0,
                help=(
                    "Number of initial training steps to keep Gumbel sampling soft "
                    "before automatically switching to hard sampling"
                ),
            )

        if "commitment_anneal_steps" not in skip:
            parser.add_argument(
                "--commitment-anneal-steps",
                type=int,
                default=0,
                help="Number of steps to linearly ramp the commitment weight",
            )

        if "compute_diversity_loss" not in skip:
            parser.add_argument(
                "--compute-diversity-loss",
                action=ActionYesNo,
                default=False,
                help="Whether to compute and report diversity loss",
            )

        if "compute_orthogonality_loss" not in skip:
            parser.add_argument(
                "--compute-orthogonality-loss",
                action=ActionYesNo,
                default=False,
                help="Whether to compute and report orthogonality loss",
            )

        if "losses_reduction" not in skip:
            parser.add_argument(
                "--losses-reduction",
                choices=["none", "mean", "sum"],
                default="mean",
                help="Reduction to apply to commitment/codebook/diversity losses",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
