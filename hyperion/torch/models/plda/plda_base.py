"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...hyper_torch_model import HyperTorchModel
from ...utils.misc import get_selfsim_tarnon, l2_norm


class PLDABase(HyperTorchModel):
    """Base class for torch PLDA backends.

    Attributes:
        x_dim: Feature dimension of the embeddings.
        mu: Global mean vector.
        num_classes: Number of enrollment reference classes.
        x_ref: Reference embeddings used for multi-class scoring.
        p_tar: Target prior probability.
        logit_ptar: Logit of :attr:`p_tar`.
        margin_multi: Multi-class margin value.
        margin_tar: Binary target margin value.
        margin_non: Binary non-target margin value.
        margin_warmup_epochs: Number of epochs used to warm up margins.
        cur_margin_multi: Current multi-class margin.
        cur_margin_tar: Current target margin.
        cur_margin_non: Current non-target margin.
        lnorm: Whether to length-normalize embeddings before scoring.
        preprocessor: Optional preprocessing module applied before scoring.
        var_floor: Lower bound used for variance-related operations.
        prec_floor: Lower bound used for precision-related operations.
        adapt_margin: Whether to adapt margins from observed scores.
        adapt_gamma: Exponential moving-average factor for margin adaptation.
        max_margin_multi: Running upper bound for the multi-class margin.
        max_margin_tar: Running upper bound for the target margin.
        max_margin_non: Running upper bound for the non-target margin.
    """

    def __init__(
        self,
        x_dim: Optional[int] = None,
        mu: Optional[torch.Tensor] = None,
        num_classes: int = 0,
        x_ref: Optional[torch.Tensor] = None,
        p_tar: float = 0.05,
        margin_multi: float = 0.3,
        margin_tar: float = 0.3,
        margin_non: float = 0.3,
        margin_warmup_epochs: int = 10,
        adapt_margin: bool = False,
        adapt_gamma: float = 0.99,
        lnorm: bool = False,
        var_floor: float = 1e-5,
        prec_floor: float = 1e-5,
        preprocessor: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the PLDA backend.

        Args:
            x_dim: Embedding dimension. Required when ``mu`` is not provided.
            mu: Optional global mean vector.
            num_classes: Number of reference classes for multi-class scoring.
            x_ref: Optional reference embeddings used for multi-class scoring.
            p_tar: Target prior probability.
            margin_multi: Multi-class margin value.
            margin_tar: Binary target margin value.
            margin_non: Binary non-target margin value.
            margin_warmup_epochs: Number of epochs used to warm up the margins.
            adapt_margin: Whether to adapt margins from observed scores.
            adapt_gamma: Exponential moving-average factor for adaptation.
            lnorm: Whether to length-normalize embeddings before scoring.
            var_floor: Lower bound for variance-like quantities.
            prec_floor: Lower bound for precision-like quantities.
            preprocessor: Optional preprocessing module applied before scoring.
        """
        super().__init__()
        if mu is None:
            assert x_dim is not None
            mu = torch.zeros((x_dim,), dtype=torch.get_default_dtype())
        else:
            mu = torch.as_tensor(mu, dtype=torch.get_default_dtype())
            x_dim = mu.shape[0]

        self.x_dim = x_dim
        self.mu = nn.Parameter(mu)

        self.p_tar = p_tar
        self.logit_ptar = math.log(p_tar / (1 - p_tar))
        self.margin_multi = margin_multi
        self.margin_tar = margin_tar
        self.margin_non = margin_non
        self.margin_warmup_epochs = margin_warmup_epochs
        if margin_warmup_epochs == 0:
            self.cur_margin_multi = margin_multi
            self.cur_margin_tar = margin_tar
            self.cur_margin_non = margin_non
        else:
            self.cur_margin_multi = 0
            self.cur_margin_tar = 0
            self.cur_margin_non = 0

        if x_ref is None:
            self.num_classes = num_classes
            if num_classes > 0:
                self.x_ref = nn.Parameter(torch.Tensor(num_classes, x_dim))
                self.x_ref.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(
                    1e5 * math.sqrt(x_dim)
                )
        else:
            x_ref = torch.as_tensor(x_ref, dtype=torch.get_default_dtype())
            self.num_classes = x_ref.shape[0]
            self.x_ref = nn.Parameter(x_ref)

        self.lnorm = lnorm
        self.preprocessor = preprocessor
        self.var_floor = var_floor
        self.prec_floor = prec_floor
        self.adapt_margin = adapt_margin
        self.adapt_gamma = adapt_gamma
        if adapt_margin:
            self.register_buffer("max_margin_multi", torch.zeros(1))
            self.register_buffer("max_margin_tar", torch.zeros(1))
            self.register_buffer("max_margin_non", torch.zeros(1))

    @staticmethod
    def l2_norm(x: torch.Tensor) -> torch.Tensor:
        """Length-normalize a batch of embeddings."""
        return math.sqrt(x.shape[-1]) * l2_norm(x)

    def __repr__(self) -> str:
        """Return the string representation."""
        return self.__str__()

    def update_margin(self, epoch: int) -> None:
        """Update the current training margins.

        Args:
            epoch: Current training epoch.
        """
        if self.margin_warmup_epochs == 0:
            return

        if self.adapt_margin:
            max_margin_multi = self.max_margin_multi
            max_margin_tar = self.max_margin_tar
            max_margin_non = self.max_margin_non
        else:
            max_margin_multi = 1
            max_margin_tar = 1
            max_margin_non = 1

        r = epoch / self.margin_warmup_epochs
        if epoch < self.margin_warmup_epochs:
            self.cur_margin_multi = r * self.margin_multi
            self.cur_margin_tar = r * self.margin_tar
            self.cur_margin_non = r * self.margin_non
            logging.info(
                ("updating plda margin_multi=%.2f " "margin_tar=%.2f margin_non=%.2f"),
                self.cur_margin_multi * max_margin_multi,
                self.cur_margin_tar * max_margin_tar,
                self.cur_margin_non * max_margin_non,
            )
        else:
            if self.cur_margin_multi != self.margin_multi:
                self.cur_margin_multi = self.margin_multi
                logging.info(
                    "updating plda margin_multi=%.2f",
                    self.cur_margin_multi * max_margin_multi,
                )
            if self.cur_margin_tar != self.margin_tar:
                self.cur_margin_tar = self.margin_tar
                logging.info(
                    "updating plda margin_tar=%.2f",
                    self.cur_margin_tar * max_margin_tar,
                )
            if self.cur_margin_non != self.margin_non:
                self.cur_margin_non = self.margin_non
                logging.info(
                    "updating plda margin_non=%.2f",
                    self.cur_margin_non * max_margin_non,
                )

        if self.adapt_margin:
            logging.info(
                ("current plda margin_multi=%.2f " "margin_tar=%.2f margin_non=%.2f"),
                self.cur_margin_multi * max_margin_multi,
                self.cur_margin_tar * max_margin_tar,
                self.cur_margin_non * max_margin_non,
            )

    def _adapt_margin_multi(self, llr: torch.Tensor, llr_tar: torch.Tensor) -> None:
        """Update the running multi-class margin bound.

        Args:
            llr: Full multi-class score matrix.
            llr_tar: Scores corresponding to the target class for each row.
        """
        tar_avg = torch.mean(llr_tar).detach()
        all_avg = torch.mean(llr).detach()
        n = llr.shape[0] * llr.shape[1]
        ntar = llr.shape[0]
        nnon = n - ntar
        non_avg = n / nnon * all_avg - ntar / nnon * tar_avg
        margin = (tar_avg - non_avg).clamp(min=0).detach()
        self.max_margin_multi = (
            self.adapt_gamma * self.max_margin_multi + (1 - self.adapt_gamma) * margin
        ).detach()

    def _adapt_margin_bin(
        self, llr: torch.Tensor, y_tar: torch.Tensor, y_non: torch.Tensor
    ) -> None:
        """Update the running binary margin bounds.

        Args:
            llr: Binary score matrix.
            y_tar: Target-class indicator matrix.
            y_non: Non-target indicator matrix.
        """
        tar_avg = torch.mean(y_tar * llr) / torch.mean(y_tar).detach()
        non_avg = torch.mean(y_non * llr) / torch.mean(y_non).detach()
        margin_tar = (tar_avg + self.logit_ptar).clamp(min=0).detach()
        margin_non = (-self.logit_ptar - non_avg).clamp(min=0).detach()
        self.max_margin_tar = (
            self.adapt_gamma * self.max_margin_tar + (1 - self.adapt_gamma) * margin_tar
        ).detach()
        self.max_margin_non = (
            self.adapt_gamma * self.max_margin_non + (1 - self.adapt_gamma) * margin_non
        ).detach()
        # logging.info('{} {} {} {}'.format(self.max_margin_tar, self.max_margin_non, margin_tar, margin_non))

    def _apply_margin_multi(
        self, llr: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the current multi-class margin to the target entries.

        Args:
            llr: Multi-class score matrix.
            y: Optional target class indices for each row.

        Returns:
            Score matrix with the target entries adjusted by the margin.
        """
        if y is None or not self.training or self.cur_margin_multi == 0:
            return llr

        if y.device != llr.device:
            y = y.to(llr.device)

        batch_size = len(llr)
        idx_ = torch.arange(0, batch_size, dtype=torch.long, device=llr.device)
        if self.adapt_margin:
            self._adapt_margin_multi(llr, llr[idx_, y])
            margin = self.cur_margin_multi * self.max_margin_multi
        else:
            margin = self.cur_margin_multi

        llr_m = llr - margin
        llr = llr * 1
        # logging.info('llr_gt={} llr_avg={}'.format(llr[idx_,y], torch.mean(llr, dim=0)))
        llr[idx_, y] = llr_m[idx_, y]
        return llr

    def _apply_margin_bin(
        self,
        llr: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        y_bin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the current binary margins to target and non-target scores.

        Args:
            llr: Binary score matrix.
            y: Optional target labels.
            y_bin: Optional binary target matrix. When omitted, it is derived
                from ``y``.

        Returns:
            Margin-adjusted score matrix.
        """
        if (
            y is None
            and y_bin is None
            or not self.training
            or self.cur_margin_tar == 0
            and self.cur_margin_non == 0
        ):
            return llr

        if y_bin is None:
            y_bin = get_selfsim_tarnon(y)
        if y_bin.device != llr.device:
            y_bin = y_bin.to(llr.device)

        y_non = 1 - y_bin
        if self.adapt_margin:
            y_tar = y_bin - torch.eye(
                len(y_bin), dtype=torch.get_default_dtype(), device=y_bin.device
            )
            self._adapt_margin_bin(llr, y_tar, y_non)
            del y_tar
            margin_tar = self.cur_margin_tar * self.max_margin_tar
            margin_non = self.cur_margin_non * self.max_margin_non
        else:
            margin_tar = self.cur_margin_tar
            margin_non = self.cur_margin_non

        llr_m = y_bin * (llr - margin_tar) + y_non * (llr + margin_non)
        return llr_m

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_multi: bool = True,
        return_bin: bool = True,
        y_bin: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Score embeddings with the configured PLDA backend.

        Args:
            x: Input embeddings.
            y: Optional labels used when applying training margins.
            return_multi: Whether to return the multi-class score matrix.
            return_bin: Whether to return the binary score matrix.
            y_bin: Optional binary target matrix used for binary margin logic.

        Returns:
            Dictionary containing ``"multi"`` and/or ``"bin"`` score matrices.
        """
        if self.preprocessor is not None:
            x = self.preprocessor(x)

        if return_multi:
            assert self.num_classes > 0
            if return_bin:
                # t = time.time()
                llr_multi, llr_bin = self.llr_1vs1_and_self(
                    x, self.x_ref, preproc=False
                )
                # logging.info('time-multi-bin={}'.format(time.time()-t))
            else:
                llr_multi = self.llr_1vs1(x, self.x_ref, preproc=False)
        elif return_bin:
            # t = time.time()
            llr_bin = self.llr_self(x, preproc=False)
            # logging.info('time-bin={}'.format(time.time()-t))

        output = {}
        if return_multi:
            output["multi"] = self._apply_margin_multi(llr_multi, y)
        if return_bin:
            output["bin"] = self._apply_margin_bin(llr_bin, y, y_bin)
        return output

    @staticmethod
    def compute_stats_hard(
        x: torch.Tensor,
        y: torch.Tensor,
        order: int = 2,
        sample_weight: Optional[torch.Tensor] = None,
        scale_factor: float = 1,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute hard-assignment sufficient statistics.

        Args:
            x: Samples with shape ``(num_samples, x_dim)``.
            y: Integer class labels for each sample.
            order: If ``2``, return only first-order stats. Any other value also
                returns second-order stats.
            sample_weight: Optional sample weights.
            scale_factor: Optional scale factor applied to the accumulated stats.

        Returns:
            ``(N, F)`` when ``order == 2``; otherwise ``(N, F, S)``.
        """
        if y.device != x.device:
            y = y.to(x.device)
        if sample_weight is not None and sample_weight.device != x.device:
            sample_weight = sample_weight.to(x.device)

        x_dim = x.shape[1]
        num_classes = int(torch.max(y).item()) + 1
        N = torch.zeros((num_classes,), dtype=x.dtype, device=x.device)
        F = torch.zeros((num_classes, x_dim), dtype=x.dtype, device=x.device)
        if sample_weight is not None:
            wx = sample_weight[:, None] * x
        else:
            wx = x

        for i in range(num_classes):
            idx = y == i
            if sample_weight is None:
                N[i] = torch.sum(idx).float()
                F[i] = torch.sum(x[idx], dim=0)
            else:
                N[i] = torch.sum(sample_weight[idx])
                F[i] = torch.sum(wx[idx], dim=0)

        if scale_factor != 1:
            N *= scale_factor
            F *= scale_factor

        if order == 2:
            return N, F

        S = torch.matmul(x.T, wx)
        if scale_factor != 1:
            S *= scale_factor

        return N, F, S

    def llr_Nvs1(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: Optional[torch.Tensor] = None,
        method: str = "vavg",
        preproc: bool = True,
    ) -> torch.Tensor:
        """Compute N-vs-1 log-likelihood ratios.

        Args:
            x1: Enrollment embeddings with shape ``(num_segments, x_dim)``.
            x2: Test embeddings with shape ``(num_test, x_dim)``.
            y1: Optional hard enrollment-side labels for ``x1``.
            method: Scoring strategy. Supported values are ``"vavg"``,
                ``"lnorm-vavg"``, and ``"savg"``. ``"book"`` is not
                implemented in the torch backend.
            preproc: Whether to apply the preprocessor before scoring.

        Returns:
            Score matrix with shape ``(num_enroll_sides, num_test)``.
        """
        method = getattr(method, "value", method)

        if self.preprocessor is not None and preproc:
            x1 = self.preprocessor(x1)
            x2 = self.preprocessor(x2)

        if method == "book":
            raise NotImplementedError(
                "Torch PLDA N-vs-1 scoring does not implement the 'book' method"
            )

        if method == "savg":
            if y1 is None:
                y1 = torch.arange(x1.shape[0], device=x1.device, dtype=torch.long)
            scores_1vs1 = self.llr_1vs1(x1, x2, preproc=False)
            N1, F1 = self.compute_stats_hard(scores_1vs1, y1)
            return F1 / N1.unsqueeze(-1)

        if y1 is not None:
            N1, F1 = self.compute_stats_hard(x1, y1)
            x1 = F1 / N1.unsqueeze(-1)

        if method not in ("vavg", "lnorm-vavg"):
            raise ValueError(f"wrong llr {method}")

        if self.lnorm or method == "lnorm-vavg":
            x1 = self.l2_norm(x1)
            x2 = self.l2_norm(x2)

        return self.llr_1vs1(x1, x2, preproc=False)

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Returns:
            Configuration dictionary that can be used to reconstruct the model.
        """
        config = {
            "x_dim": self.x_dim,
            "num_classes": self.num_classes,
            "p_tar": self.p_tar,
            "margin_multi": self.margin_multi,
            "margin_tar": self.margin_tar,
            "margin_non": self.margin_non,
            "margin_warmup_epochs": self.margin_warmup_epochs,
            "adapt_margin": self.adapt_margin,
            "adapt_gamma": self.adapt_gamma,
            "lnorm": self.lnorm,
            "var_floor": self.var_floor,
            "prec_floor": self.prec_floor,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
