"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

import torch
import torch.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn import Linear

from ...utils.misc import filter_func_args
from ..layers import ArcLossOutput, CosLossOutput, SubCenterArcLossOutput
from ..losses.rate_distortion import SubspaceLikeGaussianCodeRateDistortionL2
from .net_arch import NetArch


class HydraHeadType(str, Enum):
    """Enumeration of supported Hydra head types."""

    CLASSIF = "classif"

    @staticmethod
    def choices() -> List[str]:
        return [e.value for e in HydraHeadType]

    @staticmethod
    def to_class(value: "HydraHeadType") -> Type["HydraHead"]:
        """Map the head type to its corresponding class.

        Returns:
            Type[HydraHead]: The class associated with the head type.
        """
        if value == HydraHeadType.CLASSIF:
            return HydraClassifHead
        else:
            raise ValueError(f"Unsupported Hydra head type: {value}")

    @staticmethod
    def from_instance(head_instance: "HydraHead") -> "HydraHeadType":
        """Map a Hydra head class to its corresponding type.

        Returns:
            HydraHeadType: The head type associated with the class.
        """
        if isinstance(head_instance, HydraClassifHead):
            return HydraHeadType.CLASSIF
        else:
            raise ValueError(f"Unsupported Hydra head class: {type(head_instance)}")


class HydraClassifLossType(str, Enum):
    """Enumeration of supported classification loss types."""

    SOFTMAX = "softmax"
    COS_SOFTMAX = "cos-softmax"
    ARC_SOFTMAX = "arc-softmax"
    SUBCENTER_ARC_SOFTMAX = "subcenter-arc-softmax"

    @staticmethod
    def choices() -> List[str]:
        return [e.value for e in HydraClassifLossType]


@dataclass
class HydraClassifHeadOutput:
    """Output structure for Hydra classification head forward pass."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    prototype_code_rate: Optional[torch.Tensor] = None


@dataclass
class HydraRegressionHeadOutput:
    """Output structure for Hydra regression head forward pass."""

    preds: torch.Tensor
    loss: Optional[torch.Tensor] = None


class HydraHead(NetArch):
    """Base class for Hydra Heads

    Attributes:
       enable_loss: if True, the forward method computes the loss if targets are provided
       reduction: reduction method for the loss ('mean', 'sum', 'none')

    """

    def __init__(self, enable_loss: bool = True, reduction: str = "mean") -> None:
        """Initialize the base Hydra head.

        Args:
            enable_loss: When True, compute loss inside `forward` if targets exist.
            reduction: Reduction strategy to apply inside the loss module.
        """
        super().__init__()
        self.enable_loss = enable_loss
        self.reduction = reduction

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration for this head.

        Returns:
            Dict[str, Any]: Configuration values needed to rebuild the head.
        """
        return {
            **super().get_config(no_class_name=no_class_name),
            "enable_loss": self.enable_loss,
            "reduction": self.reduction,
        }

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI arguments that control the Hydra head behaviour.

        Args:
            parser: Argument parser where the options are registered.
            prefix: Optional prefix for grouping the arguments.
            skip: Optional set of argument names to omit.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip = skip or set()

        if "enable_loss" not in skip:
            parser.add_argument(
                "--enable-loss",
                action=ActionYesNo,
                default=True,
                help="if true, the forward method computes the loss if targets are provided",
            )

        if "reduction" not in skip:
            parser.add_argument(
                "--reduction",
                default="mean",
                choices=["none", "mean", "sum"],
                help="reduction method for the loss",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


class HydraClassifHead(HydraHead):
    """Classification head tailored for q-vector style networks.

    Attributes:
        in_feats: Input feature dimension.
        num_classes: Number of output classes.
        loss_type: Loss variant to use (`softmax`, `cos-softmax`, `arc-softmax`,
            or `subcenter-arc-softmax`).
        cos_scale: Scale factor applied by cosine-based losses.
        margin: Margin parameter for cosine and arc losses.
        margin_warmup_steps: Training steps used to anneal the margin from 0 to
            the target value.
        intertop_k: Number of hardest negatives penalised by the InterTopK term.
        intertop_margin: Penalty applied by the InterTopK regulariser.
        num_subcenters: Number of sub-centres when using sub-centre arc losses.
        enable_loss: Whether the module computes cross-entropy loss on forward.
        reduction: Reduction applied by the cross-entropy loss.
        label_smoothing: Label smoothing factor for the cross-entropy loss.
        enable_prototype_code_rate: Whether to compute the prototype code rate in `forward`.
        code_rate_eps: Epsilon parameter for the prototype code rate computation.
    """

    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        loss_type: HydraClassifLossType = HydraClassifLossType.ARC_SOFTMAX,
        cos_scale: float = 64.0,
        margin: float = 0.3,
        margin_warmup_steps: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        enable_loss: bool = True,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        enable_prototype_code_rate: bool = False,
        code_rate_eps: float = 0.5,
    ) -> None:
        """Build the classification head and configure its loss module.

        Args:
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            loss_type: Loss variant used to compute logits.
            cos_scale: Scale applied by cosine-based losses.
            margin: Margin applied by cosine-based losses.
            margin_warmup_steps: Training steps spent annealing the margin.
            intertop_k: Number of hardest negatives in the InterTopK penalty.
            intertop_margin: Penalty applied by the InterTopK term.
            num_subcenters: Number of sub-centres for sub-centre arc losses.
            enable_loss: When True, compute cross-entropy loss in `forward`.
            reduction: Reduction strategy for cross-entropy loss.
            label_smoothing: Label smoothing factor for cross-entropy loss.
            enable_prototype_code_rate: When True, compute the code rate of the prototypes in `forward`.
            code_rate_eps: Epsilon parameter for the prototype code rate computation.
        """
        super().__init__(enable_loss, reduction)
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_steps = margin_warmup_steps
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        self.num_subcenters = num_subcenters
        self.enable_prototype_code_rate = enable_prototype_code_rate
        self.code_rate_eps = code_rate_eps

        # output layer
        if loss_type == HydraClassifLossType.SOFTMAX:
            self.output = Linear(in_feats, num_classes)
        elif loss_type == HydraClassifLossType.COS_SOFTMAX:
            self.output = CosLossOutput(
                in_feats,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=0,
                margin_warmup_steps=margin_warmup_steps,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == HydraClassifLossType.ARC_SOFTMAX:
            self.output = ArcLossOutput(
                in_feats,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=0,
                margin_warmup_steps=margin_warmup_steps,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == HydraClassifLossType.SUBCENTER_ARC_SOFTMAX:
            self.output = SubCenterArcLossOutput(
                in_feats,
                num_classes,
                num_subcenters,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=0,
                margin_warmup_steps=margin_warmup_steps,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        if self.enable_loss:
            self.loss = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing
            )

        if self.enable_prototype_code_rate:
            self.code_rate = SubspaceLikeGaussianCodeRateDistortionL2(
                eps=code_rate_eps,
                normalize=False,
                distributed_mode="local",
            )

    @property
    def head_type(self) -> HydraHeadType:
        """Return the head type identifier for this head instance."""
        return HydraHeadType.CLASSIF

    @property
    def prototypes(self) -> torch.Tensor:
        """Return the learned prototypes (class centers) of the head."""
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return self.output.weight

        return self.output.prototypes

    @property
    def normalized_prototypes(self) -> torch.Tensor:
        """Return the learned prototypes (class centers) of the head."""
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return nn.functional.normalize(self.output.weight, p=2, dim=1)

        return self.output.prototypes

    def reconfig_or_create(
        self,
        in_feats: int,
        num_classes: Optional[int] = None,
        loss_type: HydraClassifLossType = HydraClassifLossType.ARC_SOFTMAX,
        cos_scale: float = 64.0,
        margin: float = 0.3,
        margin_warmup_steps: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        enable_loss: bool = True,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        enable_prototype_code_rate: bool = False,
        code_rate_eps: float = 0.5,
    ) -> "HydraClassifHead":
        """Reconfigure the head or create a new one when structural settings change.

        Args:
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            loss_type: Loss variant used to compute logits.
            cos_scale: Scale applied by cosine-based losses.
            margin: Margin applied by cosine-based losses.
            margin_warmup_steps: Training steps spent annealing the margin.
            intertop_k: Number of hardest negatives in the InterTopK penalty.
            intertop_margin: Penalty applied by the InterTopK term.
            num_subcenters: Number of sub-centres for sub-centre arc losses.
            enable_loss: When True, compute cross-entropy loss in `forward`.
            reduction: Reduction strategy for cross-entropy loss.
            label_smoothing: Label smoothing factor for cross-entropy loss.
            enable_prototype_code_rate: When True, compute the prototype code rate in `forward`.
            code_rate_eps: Epsilon parameter for the prototype code rate computation.
        Returns:
            HydraClassifHead: Updated head instance.
        """
        if num_classes is None:
            num_classes = self.num_classes

        old_cfg = self.get_config(no_class_name=True)
        new_cfg = {
            "in_feats": in_feats,
            "num_classes": num_classes,
            "loss_type": loss_type,
            "cos_scale": cos_scale,
            "margin": margin,
            "margin_warmup_steps": margin_warmup_steps,
            "intertop_k": intertop_k,
            "intertop_margin": intertop_margin,
            "num_subcenters": num_subcenters,
            "enable_loss": enable_loss,
            "reduction": reduction,
            "label_smoothing": label_smoothing,
            "enable_prototype_code_rate": enable_prototype_code_rate,
            "code_rate_eps": code_rate_eps,
        }

        if (
            in_feats != self.in_feats
            or num_classes != self.num_classes
            or loss_type != self.loss_type
            or num_subcenters != self.num_subcenters
        ):
            logging.info(
                "Rebuilding HydraClassifHead with configuration change old=%s new=%s",
                old_cfg,
                new_cfg,
            )
            return HydraClassifHead(
                in_feats=in_feats,
                num_classes=num_classes,
                loss_type=loss_type,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_steps=margin_warmup_steps,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
                enable_loss=enable_loss,
                reduction=reduction,
                label_smoothing=label_smoothing,
                enable_prototype_code_rate=enable_prototype_code_rate,
                code_rate_eps=code_rate_eps,
            )

        logging.info(
            "Updating HydraClassifHead with configuration change old=%s new=%s",
            old_cfg,
            new_cfg,
        )

        self.set_margin(margin)
        self.set_margin_warmup_steps(margin_warmup_steps)
        self.set_cos_scale(cos_scale)
        self.set_intertop_k(intertop_k)
        self.set_intertop_margin(intertop_margin)
        if self.enable_loss:
            if enable_loss:
                self.loss.reduction = reduction
                self.loss.label_smoothing = label_smoothing
            else:
                del self.loss
        elif enable_loss:
            self.loss = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing
            )

        if self.enable_prototype_code_rate:
            if enable_prototype_code_rate:
                self.code_rate.eps = code_rate_eps
            else:
                del self.code_rate
        elif enable_prototype_code_rate:
            self.code_rate = SubspaceLikeGaussianCodeRateDistortionL2(
                eps=code_rate_eps,
                normalize=False,
                distributed_mode="local",
            )

        self.reduction = reduction
        self.enable_loss = enable_loss
        self.enable_prototype_code_rate = enable_prototype_code_rate
        self.code_rate_eps = code_rate_eps
        return self

    def set_margin(self, margin: float) -> None:
        """Update the margin parameter for cosine-based losses.

        Args:
            margin: New margin to use.
        """
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return

        self.margin = margin
        self.output.margin = margin

    def set_margin_warmup_steps(self, margin_warmup_steps: int) -> None:
        """Update the number of training steps used to anneal the margin.

        Args:
            margin_warmup_steps: New warmup training-step count.
        """
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return

        self.margin_warmup_steps = margin_warmup_steps
        if hasattr(self.output, "margin_warmup_steps"):
            self.output.margin_warmup_steps = margin_warmup_steps
            if hasattr(self.output, "margin_warmup_epochs"):
                self.output.margin_warmup_epochs = 0
            if hasattr(self.output, "_update_on_step"):
                self.output._update_on_step = margin_warmup_steps > 0
        elif hasattr(self.output, "margin_warmup_epochs"):
            self.output.margin_warmup_epochs = margin_warmup_steps

    def set_cos_scale(self, cos_scale: float) -> None:
        """Update the scale parameter applied in cosine-based logits.

        Args:
            cos_scale: New cosine scale.
        """
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return

        self.cos_scale = cos_scale
        self.output.cos_scale = cos_scale

    def set_intertop_k(self, intertop_k: int) -> None:
        """Adjust how many hardest negatives contribute to the InterTopK penalty.

        Args:
            intertop_k: Updated InterTopK setting.
        """
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return

        self.intertop_k = intertop_k
        self.output.intertop_k = intertop_k

    def set_intertop_margin(self, intertop_margin: float) -> None:
        """Adjust the InterTopK margin applied to hardest negatives.

        Args:
            intertop_margin: Updated InterTopK penalty.
        """
        if self.loss_type == HydraClassifLossType.SOFTMAX:
            return

        self.intertop_margin = intertop_margin
        self.output.intertop_margin = intertop_margin

    def set_num_subcenters(self, num_subcenters: int) -> None:
        """Adjust the number of sub-centres for sub-centre arc losses.

        Args:
            num_subcenters: Updated number of sub-centres.
        """
        if not self.loss_type == HydraClassifLossType.SUBCENTER_ARC_SOFTMAX:
            return

        self.num_subcenters = num_subcenters
        self.output.num_subcenters = num_subcenters

    def update_margin(self, step: int) -> None:
        """Propagate the current training step to refresh the loss margin.

        Args:
            step: Current training step.
        """
        if hasattr(self.output, "update_margin"):
            self.output.update_margin(step)

    def forward(
        self,
        feats: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> HydraClassifHeadOutput:
        """Compute logits (and loss if enabled) for the supplied embeddings.

        Args:
            feats: Input embeddings to classify.
            target: Optional ground-truth class indices.
            target_mask: Optional mask to drop invalid target entries.

        Returns:
            HydraClassifHeadOutput: Logits and (optionally) cross-entropy loss.
        """

        if self.loss_type == HydraClassifLossType.SOFTMAX:
            logits = self.output(feats)
        else:
            logits = self.output(feats, target)

        if self.enable_loss and target is not None:
            with amp.autocast(device_type=logits.device.type, enabled=False):
                loss = self.compute_loss(logits.float(), target, target_mask)
        else:
            loss = None

        if self.enable_prototype_code_rate:
            code_rate = self.code_rate(self.normalized_prototypes)
        else:
            code_rate = None

        output = HydraClassifHeadOutput(
            logits=logits, loss=loss, prototype_code_rate=code_rate
        )
        return output

    def compute_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss, optionally masking invalid targets.

        Args:
            logits: Model logits.
            target: Ground-truth class indices.
            target_mask: Optional mask to ignore certain targets.

        Returns:
            torch.Tensor: Loss scalar reduced according to `reduction`.
        """
        if target_mask is not None:
            # remove invalid targets
            logits = logits[target_mask]
            target = target[target_mask]
            if target.numel() == 0:
                # Keep a differentiable zero-valued loss on the right device/dtype.
                if self.reduction == "none":
                    return logits.new_zeros((0,))
                return logits.sum() * 0.0

        loss = self.loss(logits, target)
        return loss

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Return the cosine affinity matrix between class prototypes.

        Returns:
            torch.Tensor: Pairwise cosine similarity between prototype weights.
        """
        if self.loss_type != HydraClassifLossType.SOFTMAX:
            return self.output.compute_prototype_affinity()

        kernel = self.output.weight  # (num_classes, feat_dim)
        kernel = kernel / torch.linalg.norm(kernel, 2, dim=1, keepdim=True)
        return torch.mm(kernel, kernel.transpose(0, 1))

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a configuration dictionary for checkpoint serialization.

        Returns:
            Dict[str, Any]: Configuration compatible with `from_json`.
        """

        config = {
            "in_feats": self.in_feats,
            "num_classes": self.num_classes,
            "loss_type": self.loss_type,
            "cos_scale": self.cos_scale,
            "margin": self.margin,
            "margin_warmup_steps": self.margin_warmup_steps,
            "intertop_k": self.intertop_k,
            "intertop_margin": self.intertop_margin,
            "num_subcenters": self.num_subcenters,
            "label_smoothing": self.loss.label_smoothing if self.enable_loss else 0.0,
            "enable_prototype_code_rate": self.enable_prototype_code_rate,
            "code_rate_eps": self.code_rate_eps,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments so only constructor parameters remain.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Filtered keyword arguments.
        """
        args = filter_func_args(HydraClassifHead.__init__, kwargs)
        return args

    @staticmethod
    def add_large_margin_loss_args(
        parser: ArgumentParser, skip: Optional[Set[str]] = None
    ) -> None:
        """Register CLI arguments controlling large-margin classification losses.

        Args:
            parser: Argument parser where the options are registered.
            skip: Optional set of argument names to omit.
        """
        skip = skip or set()

        if "cos_scale" not in skip:
            parser.add_argument(
                "--cos-scale", default=64, type=float, help="scale for arcface"
            )

        if "margin" not in skip:
            parser.add_argument(
                "--margin",
                default=0.3,
                type=float,
                help="margin for arcface, cosface,...",
            )

        if "margin_warmup_steps" not in skip:
            parser.add_argument(
                "--margin-warmup-steps",
                default=0,
                type=int,
                help="number of steps until we set the final margin",
            )

        if "intertop_k" not in skip:
            parser.add_argument(
                "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
            )
        if "intertop_margin" not in skip:
            parser.add_argument(
                "--intertop-margin",
                default=0.0,
                type=float,
                help="margin for InterTopK penalty",
            )

        if "num_subcenters" not in skip:
            parser.add_argument(
                "--num-subcenters",
                default=2,
                type=int,
                help="number of subcenters in subcenter losses",
            )

    @staticmethod
    def add_cross_entropy_loss_args(
        parser: ArgumentParser, skip: Optional[Set[str]] = None
    ) -> None:
        """Register CLI arguments for cross-entropy configuration.

        Args:
            parser: Argument parser where the options are registered.
            skip: Optional set of argument names to omit.
        """
        skip = skip or set()
        if "num_classes" not in skip:
            parser.add_argument(
                "--num-classes", type=int, default=None, help="number of output classes"
            )

        if "label_smoothing" not in skip:
            parser.add_argument(
                "--label-smoothing",
                default=0.0,
                type=float,
                help="label smoothing value for cross-entropy loss",
            )

    @staticmethod
    def add_prototype_code_rate_args(
        parser: ArgumentParser, skip: Optional[Set[str]] = None
    ) -> None:
        """Register CLI arguments for prototype code rate configuration.

        Args:
            parser: Argument parser where the options are registered.
            skip: Optional set of argument names to omit.
        """
        skip = skip or set()
        if "enable_prototype_code_rate" not in skip:
            parser.add_argument(
                "--enable-prototype-code-rate",
                default=False,
                action=ActionYesNo,
                help="enable the computation of the prototype code rate",
            )

        if "code_rate_eps" not in skip:
            parser.add_argument(
                "--code-rate-eps",
                default=0.5,
                type=float,
                help="epsilon parameter for the prototype code rate computation",
            )

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register all CLI arguments required to configure this head.

        Args:
            parser: Argument parser where the options are registered.
            prefix: Optional prefix for grouping the arguments.
            skip: Optional set of argument names to omit.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip = skip or set()

        if "loss_type" not in skip:
            parser.add_argument(
                "--loss-type",
                default=HydraClassifLossType.ARC_SOFTMAX.value,
                choices=HydraClassifLossType.choices(),
                help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
            )
        HydraClassifHead.add_large_margin_loss_args(parser, skip=skip)
        HydraClassifHead.add_cross_entropy_loss_args(parser, skip=skip)
        HydraClassifHead.add_prototype_code_rate_args(parser, skip=skip)
        HydraHead.add_class_args(parser, prefix=None, skip=skip)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
