"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeansVectorQuantizer(nn.Module):
    def __init__(self, num_embed, embed_dim, commitment_cost=0.25):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost

        # self.embed = nn.Embedding(num_embed, embed_dim)
        self.embed = nn.Parameter(torch.empty(num_embed, embed_dim))
        # this how it is init in DeepMind code:
        # self.embed.weight.data.uniform_(-math.sqrt(3)/math.sqrt(num_embed), math.sqrt(3)/math.sqrt(num_embed))
        # or equivalently:
        # nn.init.kaiming_uniform_(self.embed.weight, mode='fan_in', nonlinearity='linear')
        # normal seems to give a little better result, but not much, still we need to explore the best init.
        nn.init.normal_(self.embed, std=1.0)
        self._log_num_embed = math.log(num_embed)

    def forward(self, inputs):
        # inputs -> z_e in paper
        # convert inputs from BCHW -> BHWC
        inputs = inputs.transpose(1, -1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_inputs = inputs.view(-1, self.embed_dim)

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

        # z_q = self.embed(q_idx).view(input_shape)

        # Loss
        vq_loss = F.mse_loss(z_q, inputs.detach())
        commitment_loss = F.mse_loss(z_q.detach(), inputs)
        loss = vq_loss + self.commitment_cost * commitment_loss

        # this allows to backprogate the gradients as if the output were equal to z_e
        z_q = inputs + (z_q - inputs).detach()

        # compute the perplexity
        probs = torch.mean(r, dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

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
        return z_q, loss, kldiv_r, perplexity

    # def forward(self, inputs):
    #     # inputs -> z_e in paper
    #     # convert inputs from BCHW -> BHWC
    #     inputs = inputs.transpose(1,-1).contiguous()
    #     input_shape = inputs.shape

    #     # Flatten input
    #     flat_inputs = inputs.view(-1, self.embed_dim)

    #     # Calculate distances
    #     d2 = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
    #           + torch.sum(self.embed.weight**2, dim=1)
    #           - 2 * torch.matmul(flat_inputs, self.embed.weight.t()))

    #     # Encoding
    #     # quantization integer indexes
    #     q_idx = torch.argmin(d2, dim=1).unsqueeze(1)
    #     # 1 hot responsibilities
    #     r = torch.zeros(q_idx.shape[0], self.num_embed, device=inputs.device)
    #     r.scatter_(1, q_idx, 1)
    #     z_q = torch.matmul(r, self.embed.weight).view(input_shape)

    #     #z_q = self.embed(q_idx).view(input_shape)

    #     # Loss
    #     vq_loss = F.mse_loss(z_q, inputs.detach())
    #     commitment_loss = F.mse_loss(z_q.detach(), inputs)
    #     loss = vq_loss + self.commitment_cost * commitment_loss

    #     #this allows to backprogate the gradients as if the output were equal to z_e
    #     z_q = inputs + (z_q-inputs).detach()

    #     # compute the perplexity
    #     probs = torch.mean(r, dim=0)
    #     perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

    #     # compute KL divergence between r and uniform categorical prior
    #     # KL = \sum_i \log(1/(1/num_embed)) = \sum_i \log(num_embed) for i = all HxH or T elements
    #     # KL is constant so it doesn't contribute to the training
    #     # but we keep it to get a better estimation of the ELBO
    #     # in the paper they don't use it
    #     num_spatial_positions = r.size(0)/inputs.size(0)
    #     kldiv_r = self._log_num_embed * num_spatial_positions * torch.ones(
    #         (inputs.size(0),1), device=inputs.device)

    #     # convert quantized from BHWC -> BCHW
    #     z_q = z_q.transpose(1,-1).contiguous()
    #     return z_q, loss, kldiv_r, perplexity


class KMeansMultiVectorQuantizer(nn.Module):
    def __init__(self, num_groups, num_embed, embed_dim, commitment_cost=0.25):
        super().__init__()
        assert (
            embed_dim % embed_dim == 0
        ), "VQ latent channels (%d) must be multiple of num_groups (%d)" % (
            embed_dim,
            num_groups,
        )

        self.num_groups = num_groups
        self.embed_dim = embed_dim
        embed_dim_i = embed_dim // num_groups
        self.vq_layers = nn.ModuleList([])
        for i in range(num_groups):
            vq_i = KMeansVectorQuantizer(num_embed, embed_dim_i, commitment_cost)
            self.vq_layers.append(vq_i)

    @property
    def num_embed(self):
        return self.vq_layers[0].num_embed

    @property
    def commitment_cost(self):
        return self.vq_layers[0].commitment_cost

    def forward(self, inputs):
        inputs = inputs.chunk(self.num_groups, dim=1)
        z_q = []
        for i in range(self.num_groups):
            z_qi, loss_i, kldiv_ri, p_i = self.vq_layers[i](inputs[i])
            z_q.append(z_qi)
            if i == 0:
                loss = loss_i
                kldiv_r = kldiv_ri
                perplexity = p_i
            else:
                loss += loss_i
                kldiv_r += kldiv_ri
                perplexity += p_i

        z_q = torch.cat(tuple(z_q), dim=1)
        loss /= self.num_groups
        perplexity /= self.num_groups

        return z_q, loss, kldiv_r, perplexity


class EMAKMeansVectorQuantizer(nn.Module):
    def __init__(
        self, num_embed, embed_dim, commitment_cost=0.25, gamma=0.99, eps=1e-5
    ):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.gamma = gamma
        self.eps = eps

        # self.embed = nn.Embedding(num_embed, embed_dim)
        # self.embed.weight.data.normal_()
        self.register_buffer("embed", torch.empty(num_embed, embed_dim))
        nn.init.normal_(self.embed, std=1.0)

        self.register_buffer("_ema_N", torch.zeros(num_embed))
        self.register_buffer("_ema_z_acc", torch.empty(num_embed, embed_dim))
        nn.init.normal_(self._ema_z_acc, std=1.0)
        # self._ema_z_acc = nn.Parameter(torch.Tensor(num_embed, embed_dim))
        # self._ema_z_acc.data.normal_()

        self._log_num_embed = math.log(num_embed)

    def forward(self, inputs):
        # inputs -> z_e in paper
        # convert inputs from BCHW -> BHWC
        inputs = inputs.transpose(1, -1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_inputs = inputs.view(-1, self.embed_dim)

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
            N = torch.sum(r, dim=0)
            ema_N = self._ema_N * self.gamma + (1 - self.gamma) * N

            N_tot = torch.sum(ema_N)
            # Laplace smoothing
            self._ema_N = (
                (ema_N + self.eps) / (N_tot + self.num_embed * self.eps) * N_tot
            ).detach()

            z_acc = torch.matmul(r.t(), flat_inputs)
            self._ema_z_acc = (
                self.gamma * self._ema_z_acc + (1 - self.gamma) * z_acc
            ).detach()
            self.embed = (self._ema_z_acc / self._ema_N.unsqueeze(1)).detach()

        # Loss
        commitment_loss = F.mse_loss(z_q.detach(), inputs)
        loss = self.commitment_cost * commitment_loss

        # this allows to backprogate the gradients as if the output were equal to z_e
        z_q = inputs + (z_q - inputs).detach()

        # compute the perplexity
        probs = torch.mean(r, dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

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
        return z_q, loss, kldiv_r, perplexity

    # def forward(self, inputs):
    #     # inputs -> z_e in paper
    #     # convert inputs from BCHW -> BHWC
    #     inputs = inputs.transpose(1,-1).contiguous()
    #     input_shape = inputs.shape

    #     # Flatten input
    #     flat_inputs = inputs.view(-1, self.embed_dim)

    #     # Calculate distances
    #     d2 = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
    #           + torch.sum(self.embed.weight**2, dim=1)
    #           - 2 * torch.matmul(flat_inputs, self.embed.weight.t()))

    #     # Encoding
    #     # quantization integer indexes
    #     q_idx = torch.argmin(d2, dim=1).unsqueeze(1)
    #     # 1 hot responsibilities
    #     r = torch.zeros(q_idx.shape[0], self.num_embed, device=inputs.device)
    #     r.scatter_(1, q_idx, 1)
    #     z_q = torch.matmul(r, self.embed.weight).view(input_shape)

    #     # Use Exponetial Moving Average (EMA) to update the embedding vectors
    #     if self.training:
    #         N = torch.sum(r, dim=0)
    #         self._ema_N = self._ema_N * self.gamma + (1 - self.gamma) * N

    #         N_tot = torch.sum(self._ema_N.data)
    #         # Laplace smoothing
    #         self._ema_N = (self._ema_N + self.eps)/(N_tot + self.num_embed * self.eps) * N_tot

    #         z_acc = torch.matmul(r.t(), flat_inputs)
    #         self._ema_z_acc = nn.Parameter(
    #             self.gamma*self._ema_z_acc + (1 - self.gamma)*z_acc,
    #             requires_grad=False)

    #         self.embed.weight = nn.Parameter(
    #             self._ema_z_acc/self._ema_N.unsqueeze(1),
    #             requires_grad=False)

    #     # Loss
    #     commitment_loss = F.mse_loss(z_q.detach(), inputs)
    #     loss = self.commitment_cost * commitment_loss

    #     #this allows to backprogate the gradients as if the output were equal to z_e
    #     z_q = inputs + (z_q-inputs).detach()

    #     # compute the perplexity
    #     probs = torch.mean(r, dim=0)
    #     perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

    #     # compute KL divergence between r and uniform categorical prior
    #     # KL = \sum_i \log(1/(1/num_embed)) = \sum_i \log(num_embed) for i = all HxH or T elements
    #     # KL is constant so it doesn't contribute to the training
    #     # but we keep it to get a better estimation of the ELBO
    #     # in the paper they don't use it
    #     num_spatial_positions = r.size(0)/inputs.size(0)
    #     kldiv_r = self._log_num_embed * num_spatial_positions * torch.ones(
    #         (inputs.size(0),1), device=inputs.device)

    #     # convert quantized from BHWC -> BCHW
    #     z_q = z_q.transpose(1,-1).contiguous()
    #     return z_q, loss, kldiv_r, perplexity


class MultiEMAKMeansVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_groups,
        num_embed,
        embed_dim,
        commitment_cost=0.25,
        gamma=0.99,
        eps=1e-5,
    ):
        super().__init__()
        assert (
            embed_dim % embed_dim == 0
        ), "VQ latent channels (%d) must be multiple of num_groups (%d)" % (
            embed_dim,
            num_groups,
        )

        self.num_groups = num_groups
        self.embed_dim = embed_dim
        embed_dim_i = embed_dim // num_groups
        self.vq_layers = nn.ModuleList([])
        for i in range(num_groups):
            vq_i = EMAKMeansVectorQuantizer(
                num_embed, embed_dim_i, commitment_cost, gamma, eps
            )
            self.vq_layers.append(vq_i)

    @property
    def num_embed(self):
        return self.vq_layers[0].num_embed

    @property
    def commitment_cost(self):
        return self.vq_layers[0].commitment_cost

    @property
    def gamma(self):
        return self.vq_layers[0].gamma

    @property
    def eps(self):
        return self.vq_layers[0].eps

    def forward(self, inputs):
        inputs = inputs.chunk(self.num_groups, dim=1)
        z_q = []
        for i in range(self.num_groups):
            z_qi, loss_i, kldiv_ri, p_i = self.vq_layers[i](inputs[i])
            z_q.append(z_qi)
            if i == 0:
                loss = loss_i
                kldiv_r = kldiv_ri
                perplexity = p_i
            else:
                loss += loss_i
                kldiv_r += kldiv_ri
                perplexity *= p_i

        z_q = torch.cat(tuple(z_q), dim=1)
        loss /= self.num_groups

        return z_q, loss, kldiv_r, perplexity
