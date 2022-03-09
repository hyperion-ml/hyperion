"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ..utils import seq_lengths_to_mask


class VectorQuantizer(nn.Module):
    """Abstract base class for vector quantization layers.

    Attributes:
      num_embed: codebook size.
      embed_feats: feature dimension of the codebook vectors.
      project: if True, it projects the input features to the embed_feats dim.
      in_feats: input feature dimension, needed when project=True.
      in_dim: number of dimensions of the input tensor in [2,5], needed when project=True
    """

    def __init__(
        self, num_embed, embed_feats, project=True, in_feats=None, in_dim=None
    ):
        super().__init__()
        self.num_embed = num_embed
        self.embed_feats = embed_feats
        self.project = project
        self._proj = None
        if project:
            assert (
                in_feats is not None
            ), "input channels must be given to make the projection"
            assert (
                in_dim is not None
            ), "input tensor dim must be given to make the projection"
            self._proj = self._make_proj(in_feats, embed_feats, in_dim)

        elif in_feats is not None:
            assert in_feats == embed_feats, (
                "in_feats (%d) != embed_feats (%), which is required when project=False"
                % (in_feats, embed_feats)
            )
        else:
            in_feats = embed_feats

        self.in_feats = in_feats
        self.in_dim = in_dim

    def __repr__(self):
        return self.__str__()

    def _make_proj(self, in_feats, out_feats, ndims):
        """Creates the feature projection layer."""
        if ndims == 2:
            return nn.Linear(in_feats, out_feats)
        elif ndims == 3:
            return nn.Conv1d(in_feats, out_feats, kernel_size=1)
        elif ndims == 4:
            return nn.Conv2d(in_feats, out_feats, kernel_size=1)
        elif ndims == 5:
            return nn.Conv3d(in_feats, out_feats, kernel_size=1)
        else:
            raise ValueError("ndim=%d is not supported" % ndims)


class KMeansVectorQuantizer(VectorQuantizer):
    """Class for K-Means vector quantization layers,
        where codebook vectors are trained by gradient descend losses.

    Attributes:
      num_embed: codebook size.
      embed_feats: feature dimension of the codebook vectors.
      commitment_cost: weight for loss that makes input features close to the codebook vectors.
      project: if True, it projects the input features to the embed_feats dim.
      in_feats: input feature dimension, needed when project=True.
      in_dim: number of dimensions of the input tensor in [2,5], needed when project=True
    """

    def __init__(
        self,
        num_embed,
        embed_feats,
        commitment_cost=0.25,
        project=True,
        in_feats=None,
        in_dim=None,
    ):
        super().__init__(
            num_embed, embed_feats, project=project, in_feats=in_feats, in_dim=in_dim
        )

        self.commitment_cost = commitment_cost

        self.embed = nn.Parameter(torch.empty(num_embed, embed_feats))
        # this how it is init in DeepMind code:
        # self.embed.weight.data.uniform_(-math.sqrt(3)/math.sqrt(num_embed), math.sqrt(3)/math.sqrt(num_embed))
        # or equivalently:
        # nn.init.kaiming_uniform_(self.embed.weight, mode='fan_in', nonlinearity='linear')
        # normal seems to give a little better result, but not much, still we need to explore the best init.
        nn.init.normal_(self.embed, std=1.0)
        self._log_num_embed = math.log(num_embed)

    def __str__(self):
        s = (
            "{}(num_embed={}, embed_feats={}, commitment_cost={}, project={}, "
            "in_feats={}, in_dim={})"
        ).format(
            self.__class__.__name__,
            self.num_embed,
            self.embed_feats,
            self.commitment_cost,
            self.project,
            self.in_feats,
            self.in_dim,
        )
        return s

    def forward(self, inputs, lengths=None, mask=None, return_r=False):
        """Quantizes the input tensor.

        Args:
          input: input tensor 2d - 5d dimension with shape (batch, channels, ...)
          lengths: when inputs is 3d, it the length of each sequence in the batch.
                   Not used if mask is given.
          mask: indicates which elements are valid, to quantize. The elements with zero
                mask are set to 0. The mask tensor should have the same shape as the
                input tensor with the channel dimension removed, shape=(batch, ...).
          return_r: it True, it returns the responsibilities.

        Returns:
           Dictionary containing quantized vectors, vq_loss, KL(q(z)||p(z)), where q(z) is
           the distribution of posterior responsabilities and p(z) is a uniform categorical
           distribution, and the log_perplexity of the responsibilities. If return_r is True,
           it also returns the responsibilities.
        """
        # inputs -> z_e in paper
        if self.project:
            inputs = self._proj(inputs)

        if mask is None and lengths is not None:
            mask = seq_lengths_to_mask(
                lengths, inputs.size(-1), time_dim=1, dtype=inputs.dtype
            )

        # convert inputs from BCHW -> BHWC
        inputs = inputs.transpose(1, -1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_inputs = inputs.view(-1, self.embed_feats)

        # Calculate distances
        d2 = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embed ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embed.t())
        )  # (batch x time, num_embeds)

        # Encoding
        # quantization integer indexes
        q_idx = torch.argmin(d2, dim=1).unsqueeze(1)  # (batch x time, 1)
        # 1 hot responsibilities
        r = torch.zeros(q_idx.shape[0], self.num_embed, device=inputs.device)
        r.scatter_(1, q_idx, 1)  # (batch x time, num_embeds)
        z_q = torch.matmul(r, self.embed).view(input_shape)  # (batch, time, embed_dim)

        if mask is not None:
            z_q = z_q * mask
            inputs = inputs * mask

        # Loss
        vq_loss = F.mse_loss(z_q, inputs.detach())  # || z_q - sg(z) ||_2
        commitment_loss = F.mse_loss(z_q.detach(), inputs)  # || z - sg (z_q) ||_2

        loss = vq_loss + self.commitment_cost * commitment_loss
        if mask is not None:
            loss /= torch.mean(mask)

        # this allows to backprogate the gradients as if the output were equal to z_e
        z_q = inputs + (z_q - inputs).detach()

        # compute the perplexity
        if mask is None:
            probs = torch.mean(r, dim=0)
        else:
            probs = torch.mean(r[mask.flatten()], dim=0)

        log_perplexity = -torch.sum(probs * torch.log(probs + 1e-10))

        # compute KL divergence between r and uniform categorical prior
        # KL = \sum_i \log(1/(1/num_embed)) = \sum_i \log(num_embed) for i = all HxH or T elements
        # KL is constant so it doesn't contribute to the training
        # but we keep it to get a better estimation of the ELBO
        # in the paper they don't use it
        num_spatial_positions = r.size(0) / inputs.size(0)
        kldiv_r = (
            self._log_num_embed
            * num_spatial_positions
            * torch.ones((inputs.size(0), 1), device=inputs.device)
        )

        # convert quantized from BHWC -> BCHW
        z_q = z_q.transpose(1, -1).contiguous()  # (batch, embed_dim, time)
        output = {
            "z_q": z_q,
            "loss": loss,
            "kldiv_qrpr": kldiv_r,
            "log_perplexity": log_perplexity,
        }

        if return_r:
            output["r"] = r

        return output


class MultiKMeansVectorQuantizer(VectorQuantizer):
    """Class for Mulit-group K-Means vector quantization layers,
        where codebook vectors are trained by gradient descend losses.
        The input tensors are divided into groups and quantized separately.

    Attributes:
      num_groups: number of codebooks.
      num_embed: codebook size.
      embed_feats: feature dimension of the codebook vectors.
      commitment_cost: weight for loss that makes input features close to the codebook vectors.
      project: if True, it projects the input features to the embed_feats dim.
      in_feats: input feature dimension, needed when project=True.
      in_dim: number of dimensions of the input tensor in [2,5], needed when project=True
    """

    def __init__(
        self,
        num_groups,
        num_embed,
        embed_feats,
        commitment_cost=0.25,
        project=True,
        in_feats=None,
        in_dim=None,
    ):
        super().__init__(
            num_embed, embed_feats, project=project, in_feats=in_feats, in_dim=in_dim
        )

        assert (
            embed_feats % num_groups == 0
        ), "VQ latent channels (%d) must be multiple of num_groups (%d)" % (
            embed_feats,
            num_groups,
        )

        self.num_groups = num_groups
        embed_feats_i = embed_feats // num_groups
        self.vq_layers = nn.ModuleList([])
        for i in range(num_groups):
            vq_i = KMeansVectorQuantizer(
                num_embed, embed_feats_i, commitment_cost, project=False
            )
            self.vq_layers.append(vq_i)

    @property
    def commitment_cost(self):
        return self.vq_layers[0].commitment_cost

    def __str__(self):
        s = (
            "{}(num_groups={}, num_embed={}, embed_feats={}, commitment_cost={}, project={}, "
            "in_feats={}, in_dim={})"
        ).format(
            self.__class__.__name__,
            self.num_groups,
            self.num_embed,
            self.embed_feats,
            self.commitment_cost,
            self.project,
            self.in_feats,
            self.in_dim,
        )
        return s

    def forward(self, inputs, lengths=None, mask=None, return_r=False):
        """Quantizes the input tensor.

        Args:
          input: input tensor 2d - 5d dimension with shape (batch, channels, ...)
          lengths: when inputs is 3d, it the length of each sequence in the batch.
                   Not used if mask is given.
          mask: indicates which elements are valid, to quantize. The elements with zero
                mask are set to 0. The mask tensor should have the same shape as the
                input tensor with the channel dimension removed, shape=(batch, ...).
          return_r: it True, it returns the responsibilities.

        Returns:
           Dictionary containing quantized vectors, vq_loss, KL(q(z)||p(z)), where q(z) is
           the distribution of posterior responsabilities and p(z) is a uniform categorical
           distribution, and the log_perplexity of the responsibilities. If return_r is True,
           it also returns the responsibilities.
        """
        if self.project:
            inputs = self._proj(inputs)

        if mask is None and lengths is not None:
            mask = seq_lengths_to_mask(
                lengths, inputs.size(-1), time_dim=1, dtype=inputs.dtype
            )

        inputs = inputs.chunk(self.num_groups, dim=1)
        z_q = []
        r = []
        for i in range(self.num_groups):
            output_i = self.vq_layers[i](inputs[i], mask=mask, return_r=return_r)
            z_qi = output_i["z_q"]
            loss_i = output_i["loss"]
            kldiv_ri = output_i["kldiv_qrpr"]
            H_i = output_i["log_perplexity"]

            z_q.append(z_qi)
            if return_r:
                r.append(output_i["r"])

            if i == 0:
                loss = loss_i
                kldiv_r = kldiv_ri
                H = H_i
            else:
                loss += loss_i
                kldiv_r += kldiv_ri
                H += H_i

        z_q = torch.cat(tuple(z_q), dim=1)
        log_perplexity = H / self.num_groups
        output = {
            "z_q": z_q,
            "loss": loss,
            "kldiv_qrpr": kldiv_r,
            "log_perplexity": log_perplexity,
        }

        if return_r:
            output["r"] = r

        return output


class EMAKMeansVectorQuantizer(VectorQuantizer):
    """Class exponential moving average vector quantization layers,

    Attributes:
      num_embed: codebook size.
      embed_feats: feature dimension of the codebook vectors.
      commitment_cost: weight for loss that makes input features close to the codebook vectors.
      gamma: exponential average coefficient.
      eps: epsilon for Laplace smoothing of the counts.
      project: if True, it projects the input features to the embed_feats dim.
      in_feats: input feature dimension, needed when project=True.
      in_dim: number of dimensions of the input tensor in [2,5], needed when project=True
    """

    def __init__(
        self,
        num_embed,
        embed_feats,
        commitment_cost=0.25,
        gamma=0.99,
        eps=1e-5,
        project=True,
        in_feats=None,
        in_dim=None,
    ):
        super().__init__(
            num_embed, embed_feats, project=project, in_feats=in_feats, in_dim=in_dim
        )

        self.num_embed = num_embed
        self.embed_feats = embed_feats
        self.commitment_cost = commitment_cost
        self.gamma = gamma
        self.eps = eps

        self.register_buffer("embed", torch.empty(num_embed, embed_feats))
        nn.init.normal_(self.embed, std=1.0)

        self.register_buffer("_ema_N", torch.zeros(num_embed))
        self.register_buffer("_ema_z_acc", torch.empty(num_embed, embed_feats))
        nn.init.normal_(self._ema_z_acc, std=1.0)

        self._log_num_embed = math.log(num_embed)

    def __str__(self):
        s = (
            "{}(num_embed={}, embed_feats={}, commitment_cost={}, "
            "gamma={}, eps={} project={}, in_feats={}, in_dim={})"
        ).format(
            self.__class__.__name__,
            self.num_embed,
            self.embed_feats,
            self.commitment_cost,
            self.gamma,
            self.eps,
            self.project,
            self.in_feats,
            self.in_dim,
        )
        return s

    def forward(self, inputs, lengths=None, mask=None, return_r=False):
        """Quantizes the input tensor. In training phase, it also
            updates the codebooks by EMA.

        Args:
          input: input tensor 2d - 5d dimension with shape (batch, channels, ...)
          lengths: when inputs is 3d, it the length of each sequence in the batch.
                   Not used if mask is given.
          mask: indicates which elements are valid, to quantize. The elements with zero
                mask are set to 0. The mask tensor should have the same shape as the
                input tensor with the channel dimension removed, shape=(batch, ...).
          return_r: it True, it returns the responsibilities.

        Returns:
           Dictionary containing quantized vectors, vq_loss, KL(q(z)||p(z)), where q(z) is
           the distribution of posterior responsabilities and p(z) is a uniform categorical
           distribution, and the log_perplexity of the responsibilities. If return_r is True,
           it also returns the responsibilities.
        """
        # inputs -> z_e in paper
        if self.project:
            inputs = self._proj(inputs)

        if mask is None and lengths is not None:
            mask = seq_lengths_to_mask(
                lengths, inputs.size(-1), time_dim=1, dtype=inputs.dtype
            )

        # convert inputs from BCHW -> BHWC
        inputs = inputs.transpose(1, -1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_inputs = inputs.view(-1, self.embed_feats)

        # Calculate distances
        d2 = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embed ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embed.t())
        )

        # Encoding
        # quantization integer indexes
        q_idx = torch.argmin(d2, dim=1).unsqueeze(1)
        # 1 hot responsibilities
        r = torch.zeros(q_idx.shape[0], self.num_embed, device=inputs.device)
        r.scatter_(1, q_idx, 1)
        z_q = torch.matmul(r, self.embed).view(input_shape)

        # Use Exponetial Moving Average (EMA) to update the embedding vectors
        if self.training:
            if mask is not None:
                flat_mask = mask.flatten()
                r = r[flat_mask]
                flat_inputs = flat_inputs[flat_mask]

            N = torch.sum(r, dim=0)
            # required to sync gpus in DDP
            if dist.is_initialized():
                dist.all_reduce(N, op=dist.ReduceOp.SUM)

            ema_N = self._ema_N * self.gamma + (1 - self.gamma) * N

            N_tot = torch.sum(ema_N)
            # Laplace smoothing
            self._ema_N = (
                (ema_N + self.eps) / (N_tot + self.num_embed * self.eps) * N_tot
            ).detach()

            z_acc = torch.matmul(r.t(), flat_inputs)
            # required to sync gpus in DDP
            if dist.is_initialized():
                dist.all_reduce(z_acc, op=dist.ReduceOp.SUM)
            self._ema_z_acc = (
                self.gamma * self._ema_z_acc + (1 - self.gamma) * z_acc
            ).detach()
            self.embed = (self._ema_z_acc / self._ema_N.unsqueeze(1)).detach()

        if mask is not None:
            z_q = z_q * mask
            inputs = inputs * mask
        # Loss
        commitment_loss = F.mse_loss(z_q.detach(), inputs)
        loss = self.commitment_cost * commitment_loss
        if mask is not None:
            loss /= torch.mean(mask)

        # this allows to backprogate the gradients as if the output were equal to z_e
        z_q = inputs + (z_q - inputs).detach()

        # compute the perplexity
        if mask is None:
            probs = torch.mean(r, dim=0)
        else:
            probs = torch.mean(r[mask.flatten()], dim=0)

        log_perplexity = -torch.sum(probs * torch.log(probs + 1e-10))

        # compute KL divergence between r and uniform categorical prior
        # KL = \sum_i \log(1/(1/num_embed)) = \sum_i \log(num_embed) for i = all HxH or T elements
        # KL is constant so it doesn't contribute to the training
        # but we keep it to get a better estimation of the ELBO
        # in the paper they don't use it
        num_spatial_positions = r.size(0) / inputs.size(0)
        kldiv_r = (
            self._log_num_embed
            * num_spatial_positions
            * torch.ones((inputs.size(0), 1), device=inputs.device)
        )

        # convert quantized from BHWC -> BCHW
        z_q = z_q.transpose(1, -1).contiguous()
        output = {
            "z_q": z_q,
            "loss": loss,
            "kldiv_qrpr": kldiv_r,
            "log_perplexity": log_perplexity,
        }

        if return_r:
            output["r"] = r

        return output


class MultiEMAKMeansVectorQuantizer(VectorQuantizer):
    """Class for Mulit-group exponential moving average vector quantization layers,
        where codebook vectors are trained by gradient descend losses.
        The input tensors are divided into groups and quantized separately.

    Attributes:
      num_groups: number of codebooks.
      num_embed: codebook size.
      embed_feats: feature dimension of the codebook vectors.
      commitment_cost: weight for loss that makes input features close to the codebook vectors.
      gamma: exponential average coefficient.
      eps: epsilon for Laplace smoothing of the counts.
      project: if True, it projects the input features to the embed_feats dim.
      in_feats: input feature dimension, needed when project=True.
      in_dim: number of dimensions of the input tensor in [2,5], needed when project=True
    """

    def __init__(
        self,
        num_groups,
        num_embed,
        embed_feats,
        commitment_cost=0.25,
        gamma=0.99,
        eps=1e-5,
        project=True,
        in_feats=None,
        in_dim=None,
    ):
        super().__init__(
            num_embed, embed_feats, project=project, in_feats=in_feats, in_dim=in_dim
        )

        assert (
            embed_feats % embed_feats == 0
        ), "VQ latent channels (%d) must be multiple of num_groups (%d)" % (
            embed_feats,
            num_groups,
        )

        self.num_groups = num_groups
        embed_feats_i = embed_feats // num_groups
        self.vq_layers = nn.ModuleList([])
        for i in range(num_groups):
            vq_i = EMAKMeansVectorQuantizer(
                num_embed, embed_feats_i, commitment_cost, gamma, eps, project=False
            )
            self.vq_layers.append(vq_i)

    @property
    def commitment_cost(self):
        return self.vq_layers[0].commitment_cost

    @property
    def gamma(self):
        return self.vq_layers[0].gamma

    @property
    def eps(self):
        return self.vq_layers[0].eps

    def __str__(self):
        s = (
            "{}(num_groups={}, num_embed={}, embed_feats={}, commitment_cost={}, "
            "gamma={}, eps={} project={}, in_feats={}, in_dim={})"
        ).format(
            self.__class__.__name__,
            self.num_groups,
            self.num_embed,
            self.embed_feats,
            self.commitment_cost,
            self.gamma,
            self.eps,
            self.project,
            self.in_feats,
            self.in_dim,
        )
        return s

    def forward(self, inputs, lengths=None, mask=None, return_r=False):
        """Quantizes the input tensor.

        Args:
          input: input tensor 2d - 5d dimension with shape=(batch, channels, ...)
          lengths: when inputs is 3d, it the length of each sequence in the batch.
                   Not used if mask is given.
          mask: indicates which elements are valid, to quantize. The elements with zero
                mask are set to 0. The mask tensor should have the same shape as the
                input tensor with the channel dimension removed, shape=(batch, ...).
          return_r: it True, it returns the responsibilities.

        Returns:
           Dictionary containing quantized vectors, vq_loss, KL(q(z)||p(z)), where q(z) is
           the distribution of posterior responsabilities and p(z) is a uniform categorical
           distribution, and the log_perplexity of the responsibilities. If return_r is True,
           it also returns the responsibilities.
        """
        if self.project:
            inputs = self._proj(inputs)

        if mask is None and lengths is not None:
            mask = seq_lengths_to_mask(
                lengths, inputs.size(-1), time_dim=1, dtype=inputs.dtype
            )

        inputs = inputs.chunk(self.num_groups, dim=1)
        z_q = []
        r = []
        for i in range(self.num_groups):
            output_i = self.vq_layers[i](inputs[i], mask=mask)
            z_qi = output_i["z_q"]
            loss_i = output_i["loss"]
            kldiv_ri = output_i["kldiv_qrpr"]
            H_i = output_i["log_perplexity"]

            z_q.append(z_qi)
            if return_r:
                r.append(output_i["r"])

            if i == 0:
                loss = loss_i
                kldiv_r = kldiv_ri
                H = H_i
            else:
                loss += loss_i
                kldiv_r += kldiv_ri
                H += H_i

        z_q = torch.cat(tuple(z_q), dim=1)
        loss /= self.num_groups
        log_perplexity = H / self.num_groups
        output = {
            "z_q": z_q,
            "loss": loss,
            "kldiv_qrpr": kldiv_r,
            "log_perplexity": log_perplexity,
        }

        if return_r:
            output["r"] = r

        return output
