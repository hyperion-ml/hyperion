from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch import Tensor

from ...utils.misc import filter_func_args


class SubspaceLikeGaussianCodeRateDistortionL2(nn.Module):
    """
    Computes the information-theoretic coding rate distortion loss.
        Yi Ma, Harm Derksen, Wei Hong, and John Wright. Segmentation of multivariate
            mixed data via lossy data coding and compression. IEEE Transactions on Pattern
            Analysis and Machine Intelligence, 29(9):1546–1562, 2007.

    - SubspaceLikeGaussian: Assumes features form low-dimensional planes + noise.
    - CodeRateDistortion: Uses Rate-Distortion theory (logdet) instead of Shannon entropy.
    - L2: Rooted in an L2-norm squared distortion error framework.

    Attributes:
        eps: Distortion tolerance parameter in the coding-rate formula.
        reduction: Reduction applied to the per-batch rates.
        jitter: Diagonal stabilization added to the identity term.
        gamma_1: Denominator scaling factor in the final coding rate.
        gamma_2: Numerator scaling factor applied inside the log-determinant.
        normalize: Whether to L2-normalize the input vectors before computing
            the rate.
        distributed_mode: How rank-2 inputs are handled in distributed training.
    """

    def __init__(
        self,
        eps: float = 0.5,
        reduction: Literal["mean", "sum", "none"] = "mean",
        jitter: float = 1e-6,
        gamma_1: float = 1.0,
        gamma_2: float = 1.0,
        normalize: bool = True,
        distributed_mode: Literal["local", "global_data"] = "global_data",
    ) -> None:
        """
        Args:
            eps: Distortion tolerance parameter in the coding-rate formula.
            reduction: Reduction applied to the per-batch rates.
            jitter: Diagonal stabilization added to the identity term.
            gamma_1: Denominator scaling factor in the final coding rate.
            gamma_2: Numerator scaling factor applied inside the log-determinant.
            normalize: If True, L2-normalizes the input vectors before computing
                the rate.
            distributed_mode: ``"local"`` computes the rate from the local tensor only.
                ``"global_data"`` synchronizes rank-2 inputs across GPUs by summing
                ``z^T z`` and the sample count ``m`` across ranks before computing the rate.
        """
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if distributed_mode not in ["local", "global_data"]:
            raise ValueError("distributed_mode must be 'local' or 'global_data'")

        self.eps = eps
        self.reduction = reduction
        self.jitter = jitter
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.normalize = normalize
        self.distributed_mode = distributed_mode

    def _safe_logdet_cholesky(self, matrix: Tensor) -> Tensor:
        """
        Computes the log-determinant of a Symmetric Positive-Definite matrix.
        Forces the fast, symmetric Cholesky factorization path.
        Supports multi-dimensional batch tensors.

        Args:
            matrix: Symmetric positive-definite matrix with shape (..., n, n).

        Returns:
            Log-determinant values with shape matching the batch dimensions.
        """
        # EXPLICIT CHOLESKY PATH
        try:
            L = torch.linalg.cholesky(matrix)
            diag = torch.diagonal(L, dim1=-2, dim2=-1)
            logdet = 2.0 * torch.sum(torch.log2(diag), dim=-1)
            return logdet
        except torch.linalg.LinAlgError:
            # Emergency SVD Fallback if Cholesky encounters structural truncation anomalies
            _, S, _ = torch.linalg.svd(matrix)
            return torch.sum(torch.log2(S), dim=-1)

    def apply_reduction(self, rates: Tensor) -> Tensor:
        """
        Reduces rates over all batch/outer dimensions.

        Args:
            rates: Tensor of per-example or batched coding rates.

        Returns:
            Reduced tensor according to ``self.reduction``.
        """
        if self.reduction == "none" or rates.dim() == 0:
            return rates
        elif self.reduction == "mean":
            return torch.mean(rates)
        elif self.reduction == "sum":
            return torch.sum(rates)

    def _use_distributed_global_data(self, z: Tensor) -> bool:
        return (
            self.distributed_mode == "global_data"
            and z.dim() == 2
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _compute_global_covariance_distributed(self, z: Tensor) -> tuple[Tensor, int]:
        """
        Computes the global uncentered covariance term for a rank-2 tensor distributed
        across GPUs along the sample axis.

        Args:
            z: Local tensor of shape (m_local, d).

        Returns:
            Tuple containing the globally aggregated ``z^T z`` tensor of shape
            (d, d) and the total sample count across all ranks.
        """
        global_cov = torch.matmul(z.transpose(-1, -2), z)
        dist.all_reduce(global_cov, op=dist.ReduceOp.SUM)

        m = torch.tensor([z.shape[0]], device=z.device, dtype=torch.long)
        dist.all_reduce(m, op=dist.ReduceOp.SUM)
        return global_cov, int(m.item())

    def _get_global_unique_classes(self, labels: Tensor) -> Tensor:
        """
        Computes the globally consistent set of unique class labels across all ranks.

        Args:
            labels: Local 1D tensor of class labels.

        Returns:
            Sorted tensor containing the union of labels across all ranks.
        """
        local_unique = torch.unique(labels)
        if not (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        ):
            return local_unique

        gathered_labels = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_labels, local_unique.detach().cpu().tolist())

        merged = sorted(
            {label for rank_labels in gathered_labels for label in rank_labels}
        )
        return torch.tensor(merged, device=labels.device, dtype=labels.dtype)

    def forward(self, z: Tensor) -> Tensor:
        """
        Computes the Coding Rate (R) of the given feature matrix.

        z Shape: (..., m, d) where:
          - d (last dimension) is the vector dimension (e.g., 192).
          - m (second to last) is the number of samples.
          - ... represents arbitrary outer batch dimensions (e.g., b).

        Args:
            z: Input feature tensor with shape (..., m, d).

        Returns:
            Coding rate tensor after applying the configured reduction.
        """
        # 1. PRECISION UPCAST: Cast features to Float32 BEFORE computing the covariance matrix
        # This preserves crucial bits of precision during the massive matrix multiplication.
        z = z.float()
        if self.normalize:
            z = nn.functional.normalize(z, dim=-1)
        d = z.shape[-1]

        if self._use_distributed_global_data(z):
            global_cov, m = self._compute_global_covariance_distributed(z)
            eye_dim = d
        else:
            m = z.shape[-2]

            # 2. Compute uncentered Gram/Covariance matrix: Z_transpose * Z
            z_transposed = z.transpose(-1, -2)  # (..., d, m)
            if d <= m:
                global_cov = torch.matmul(z_transposed, z)  # (..., d, d)
                eye_dim = d
            else:
                # We can do this by using the fact that logdet(I + c * Z^T Z) = logdet(I + c * Z Z^T)
                global_cov = torch.matmul(z, z_transposed)  # (..., m, m)
                eye_dim = m

        # 3. Calculate the scalar coefficient: d / (m * eps^2)
        scalar = self.gamma_2 * d / (m * (self.eps**2))

        # 4. Construct the core matrix: I * (1.0 + jitter) + scalar * Covariance
        # This blends the MCR2 identity requirement and the Cholesky safety net into a single allocation.
        identity_scale = 1.0 + self.jitter
        identity_block = (
            torch.eye(eye_dim, device=z.device).expand_as(global_cov) * identity_scale
        )

        matrices_to_det = identity_block + (scalar * global_cov)

        # 5. Calculate: 0.5 * logdet(I_stable + c * Z^T * Z)
        R = (0.5 / self.gamma_1) * self._safe_logdet_cholesky(matrices_to_det)

        # 6. Apply reduction across all dimensions except the last two
        return self.apply_reduction(R)

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(
            SubspaceLikeGaussianCodeRateDistortionL2.__init__, kwargs
        )

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--eps",
            default=0.5,
            type=float,
            help="Distortion tolerance parameter in the coding-rate formula",
        )
        # parser.add_argument(
        #     "--reduction",
        #     default="mean",
        #     type=str,
        #     choices=["mean", "sum", "none"],
        #     help="Reduction applied to the per-batch rates",
        # )
        parser.add_argument(
            "--jitter",
            default=1e-6,
            type=float,
            help="Diagonal stabilization added to the identity term",
        )
        parser.add_argument(
            "--gamma-1",
            default=1.0,
            type=float,
            help="Denominator scaling factor in the final coding rate",
        )
        parser.add_argument(
            "--gamma-2",
            default=1.0,
            type=float,
            help="Numerator scaling factor applied inside the log-determinant",
        )
        parser.add_argument(
            "--normalize",
            default=True,
            action=ActionYesNo,
            help="Whether to L2-normalize the input vectors before computing the rate",
        )
        # parser.add_argument(
        #     "--distributed-mode",
        #     default="local",
        #     type=str,
        #     choices=["local", "global_data"],
        #     help="How rank-2 inputs are handled in distributed training",
        # )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


class CategoricalSubspaceLikeGaussianCodeRateDistortionL2(
    SubspaceLikeGaussianCodeRateDistortionL2
):
    """
    Computes categorical coding rates by partitioning samples by class, evaluating
    one coding-rate term per class, and combining them using class proportions.

    This variant extends ``SubspaceLikeGaussianCodeRateDistortionL2`` for labeled
    data where each class defines its own subspace-like distribution.

    Attributes:
        eps: Distortion tolerance parameter in the coding-rate formula.
        jitter: Diagonal stabilization added to the identity term.
        normalize: Whether to L2-normalize the input vectors before computing
            class-wise rates.
        distributed_mode: How rank-2 inputs are handled in distributed training.
    """

    def __init__(
        self,
        eps: float = 0.5,
        jitter: float = 1e-6,
        normalize: bool = True,
        distributed_mode: Literal["local", "global_data"] = "global_data",
    ) -> None:
        """
        Args:
            eps: Distortion tolerance parameter in the coding-rate formula.
            jitter: Diagonal stabilization added to the identity term.
            normalize: If True, L2-normalizes the input vectors before computing
                class-wise rates.
            distributed_mode: ``"local"`` computes class-wise rates from local tensors
                only. ``"global_data"`` synchronizes rank-2 inputs across GPUs by
                summing class-wise covariance terms and sample counts across ranks
                before computing rates.
        """
        super().__init__(
            eps=eps,
            jitter=jitter,
            normalize=normalize,
            distributed_mode=distributed_mode,
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """
        Computes the class-wise coding rates and combines them according to class
        proportions.

        Args:
            z: Input feature tensor with shape (..., m, d).
            labels: Class labels associated with the samples in ``z`` with shape
                (..., m).

        Returns:
            Weighted average coding rate across classes.
        """
        # 1. PRECISION UPCAST: Cast features to Float32 BEFORE computing the covariance matrix
        z = z.float()
        if self.normalize:
            z = nn.functional.normalize(z, dim=-1)
        d = z.shape[-1]

        # 2. Flatten outer dimensions for easier class-wise processing
        z_flat = z.reshape(-1, d)  # (m, d)
        labels_flat = labels.flatten()  # (m,)

        # 3. Compute class-wise covariance matrices and sample counts
        if self._use_distributed_global_data(z):
            unique_classes = self._get_global_unique_classes(labels_flat)
        else:
            unique_classes = torch.unique(labels_flat)
        class_covs = []
        m_js = []
        for cls in unique_classes:
            mask = labels_flat == cls  # (m,)
            z_cls = z_flat[mask]
            # (m_j, d) where m_j is the number of samples in class j

            if self._use_distributed_global_data(z):
                cov_cls, m_j = self._compute_global_covariance_distributed(z_cls)
            else:
                m_j = z_cls.shape[0]
                # 2. Compute uncentered Gram/Covariance matrix: Z_transpose * Z
                cov_cls = torch.matmul(z_cls.transpose(-1, -2), z_cls)  # (d, d)

            class_covs.append(cov_cls)
            m_js.append(m_j)

        # 3. Calculate the scalar coefficient: d / (m * eps^2)
        class_covs = torch.stack(class_covs, dim=0)  # (num_classes, d, d)
        m_js = torch.as_tensor(
            m_js, device=z.device, dtype=torch.float32
        )  # (num_classes,)
        scalars = d / (m_js * (self.eps**2))

        # 4. Construct the core matrix: I * (1.0 + jitter) + scalar * Covariance
        # This blends the MCR2 identity requirement and the Cholesky safety net into a single allocation.
        identity_scale = 1.0 + self.jitter
        identity_block = torch.eye(d, device=z.device).unsqueeze(0) * identity_scale
        matrices_to_det = identity_block + (scalars.view(-1, 1, 1) * class_covs)

        # 5. Calculate: 0.5 * logdet(I_stable + c * Z^T * Z)
        R = 0.5 * self._safe_logdet_cholesky(matrices_to_det)  # (num_classes,)

        # 6. Implement the weighted partition average: sum( (m_j / (2 * m)) * logdet_j )
        R = (m_js / m_js.sum()) * R  # Weight by class proportions
        Rc = R.sum()  # Total coding rate across classes

        return Rc

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(
            CategoricalSubspaceLikeGaussianCodeRateDistortionL2.__init__, kwargs
        )

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--eps",
            default=0.5,
            type=float,
            help="Distortion tolerance parameter in the coding-rate formula",
        )
        parser.add_argument(
            "--jitter",
            default=1e-6,
            type=float,
            help="Diagonal stabilization added to the identity term",
        )
        parser.add_argument(
            "--normalize",
            default=True,
            action=ActionYesNo,
            help="Whether to L2-normalize the input vectors before computing class-wise rates",
        )
        # parser.add_argument(
        #     "--distributed-mode",
        #     default="global_data",
        #     type=str,
        #     choices=["local", "global_data"],
        #     help="How rank-2 inputs are handled in distributed training",
        # )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


class CategoricalSubspaceLikeGaussianCodeRateDistortionL2Reduction(nn.Module):
    """
    Computes coding-rate reduction by subtracting the categorical coding rate
    from the overall coding rate.

    This module combines:
    - ``SubspaceLikeGaussianCodeRateDistortionL2`` to compute the overall rate.
    - ``CategoricalSubspaceLikeGaussianCodeRateDistortionL2`` to compute the
      class-weighted categorical rate.

    Attributes:
        categorical_rate_loss: Categorical coding-rate module used to compute
            the class-weighted rate term.
        normalize: Whether to L2-normalize the input vectors before computing
            the rate terms.
        overall_rate_loss: Coding-rate module used to compute the overall rate
            term.
    """

    def __init__(
        self,
        eps: float = 0.5,
        jitter: float = 1e-6,
        gamma_1: float = 1.0,
        gamma_2: float = 1.0,
        normalize: bool = True,
        distributed_mode: Literal["local", "global_data"] = "global_data",
    ) -> None:
        """
        Args:
            eps: Distortion tolerance parameter in the coding-rate formula.
            jitter: Diagonal stabilization added to the identity term.
            gamma_1: Denominator scaling factor in the overall coding-rate term.
            gamma_2: Numerator scaling factor applied inside the overall
                log-determinant term.
            normalize: If True, L2-normalizes the input vectors before computing
                the rate terms.
            distributed_mode: ``"local"`` computes rates from local tensors only.
                ``"global_data"`` synchronizes rank-2 inputs across GPUs when
                computing the global-data variants of the rate terms.
        """
        super().__init__()
        self.normalize = normalize
        self.categorical_rate_loss = (
            CategoricalSubspaceLikeGaussianCodeRateDistortionL2(
                eps=eps,
                jitter=jitter,
                normalize=False,
                distributed_mode=distributed_mode,
            )
        )
        self.overall_rate_loss = SubspaceLikeGaussianCodeRateDistortionL2(
            eps=eps,
            jitter=jitter,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            normalize=False,
            distributed_mode=distributed_mode,
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """
        Computes the coding-rate reduction ``R - R_c``.

        Args:
            z: Input feature tensor with shape (..., m, d).
            labels: Class labels associated with the samples in ``z`` with shape
                (..., m).

        Returns:
            Scalar coding-rate reduction value.
        """
        d = z.shape[-1]
        z = z.reshape(-1, d)
        if self.normalize:
            z = nn.functional.normalize(z.float(), p=2, dim=-1)

        R = self.overall_rate_loss(z)
        Rc = self.categorical_rate_loss(z, labels)
        delta_R = R - Rc
        return delta_R

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(
            CategoricalSubspaceLikeGaussianCodeRateDistortionL2Reduction.__init__,
            kwargs,
        )

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--eps",
            default=0.5,
            type=float,
            help="Distortion tolerance parameter in the coding-rate formula",
        )
        parser.add_argument(
            "--jitter",
            default=1e-6,
            type=float,
            help="Diagonal stabilization added to the identity term",
        )
        parser.add_argument(
            "--gamma-1",
            default=1.0,
            type=float,
            help="Denominator scaling factor in the overall coding-rate term",
        )
        parser.add_argument(
            "--gamma-2",
            default=1.0,
            type=float,
            help="Numerator scaling factor applied inside the overall log-determinant term",
        )
        parser.add_argument(
            "--normalize",
            default=True,
            action=ActionYesNo,
            help="Whether to L2-normalize the input vectors before computing the rate terms",
        )
        # parser.add_argument(
        #     "--distributed-mode",
        #     default="global_data",
        #     type=str,
        #     choices=["local", "global_data"],
        #     help="How rank-2 inputs are handled in distributed training",
        # )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
