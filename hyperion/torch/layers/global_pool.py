"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnf

SQRT_EPS = 1e-5


def _conv1(in_channels, out_channels, bias=False):
    """point-wise convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


class _GlobalPool1d(nn.Module):
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.size_multiplier = 1

    def _standarize_weights(self, weights, ndims):

        if weights.dim() == ndims:
            return weights

        assert weights.dim() == 2
        shape = ndims * [1]
        shape[0] = weights.shape[0]
        shape[self.dim] = weights.shape[1]
        return weights.view(tuple(shape))

    def get_config(self):
        config = {"dim": self.dim, "keepdim": self.keepdim}
        return config

    def forward_slidwin(self, x, win_length, win_shift):
        raise NotImplementedError()

    def _slidwin_pad(self, x, win_length, win_shift, snip_edges):

        if snip_edges:
            num_frames = int(
                math.floor((x.size(-1) - win_length + win_shift) / win_shift)
            )
            return nnf.pad(x, (1, 0), mode="constant"), num_frames

        assert (
            win_length >= win_shift
        ), "if win_length < win_shift snip-edges should be false"

        num_frames = int(round(x.size(-1) / win_shift))
        len_x = (num_frames - 1) * win_shift + win_length
        dlen_x = round(len_x - x.size(-1))
        pad_left = int(math.floor((win_length - win_shift) / 2))
        pad_right = int(dlen_x - pad_left)

        return nnf.pad(x, (pad_left + 1, pad_right), mode="reflect"), num_frames


class GlobalAvgPool1d(_GlobalPool1d):
    """Global average pooling in 1d

    Attributes:
       dim: pooling dimension
       keepdim: it True keeps the same number of dimensions after pooling

    """

    def __init__(self, dim=-1, keepdim=False):
        super().__init__(dim, keepdim)

    def forward(self, x, weights=None):
        if weights is None:
            y = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            return y

        weights = self._standarize_weights(weights, x.dim())

        xbar = torch.mean(weights * x, dim=self.dim, keepdim=self.keepdim)
        wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        return xbar / wbar

    def forward_slidwin(self, x, win_length, win_shift, snip_edges=False):
        if isinstance(win_shift, int) and isinstance(win_length, int):
            return self._forward_slidwin_int(
                x, win_length, win_shift, snip_edges=snip_edges
            )

        # the window length and/or shift are floats
        return self._forward_slidwin_float(
            x, win_length, win_shift, snip_edges=snip_edges
        )

    def _pre_slidwin(self, x, win_length, win_shift, snip_edges):
        if self.dim != -1:
            x = x.transpose(self.dim, -1)

        x, num_frames = self._slidwin_pad(x, win_length, win_shift, snip_edges)
        out_shape = *x.shape[:-1], num_frames
        c_x = torch.cumsum(x, dim=-1).view(-1, x.shape[-1])
        return c_x, out_shape

    def _post_slidwin(self, m_x, x_shape):
        m_x = m_x.view(x_shape)

        if self.dim != -1:
            m_x = m_x.transpose(self.dim, -1).contiguous()

        return m_x

    def _forward_slidwin_int(self, x, win_length, win_shift, snip_edges):

        c_x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)

        m_x = (c_x[:, win_shift:] - c_x[:, :-win_shift]) / win_length

        m_x = self._post_slid_win(m_x, out_shape)
        return m_x

    def _forward_slidwin_float(self, x, win_length, win_shift, snip_edges):
        c_x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)

        num_frames = out_shape[-1]
        m_x = torch.zeros(
            (c_x.shape[0], num_frames), dtype=c_x.dtype, device=c_x.device
        )
        k = 0
        for i in range(num_frames):
            k1 = int(round(k))
            k2 = int(round(k + win_length))
            m_x[:, i] = (c_x[:, k2] - c_x[:, k1]) / (k2 - k1)
            k += win_shift

        m_x = self._post_slid_win(m_x, out_shape)
        return m_x


class GlobalMeanStdPool1d(_GlobalPool1d):
    """Global mean + standard deviation pooling in 1d

    Attributes:
       dim: pooling dimension
       keepdim: it True keeps the same number of dimensions after pooling

    """

    def __init__(self, dim=-1, keepdim=False):
        super().__init__(dim, keepdim)
        self.size_multiplier = 2

    def forward(self, x, weights=None):
        if weights is None:
            mu = torch.mean(x, dim=self.dim, keepdim=True)
            delta = x - mu
            mu.squeeze_(dim=self.dim)

            # this can produce slightly negative variance when relu6 saturates in all time steps
            # add 1e-5 for stability
            s = torch.sqrt(
                torch.mean(delta ** 2, dim=self.dim, keepdim=False).clamp(min=SQRT_EPS)
            )

            mus = torch.cat((mu, s), dim=1)
            if self.keepdim:
                mus.unsqueeze_(dim=self.dim)

            return mus

        weights = self._standarize_weights(weights, x.dim())
        xbar = torch.mean(weights * x, dim=self.dim, keepdim=True)
        wbar = torch.mean(weights, dim=self.dim, keepdim=True)
        mu = xbar / wbar
        delta = x - mu
        var = torch.mean(weights * delta ** 2, dim=self.dim, keepdim=True) / wbar
        s = torch.sqrt(var.clamp(min=SQRT_EPS))
        mu = mu.squeeze(self.dim)
        s = s.squeeze(self.dim)
        mus = torch.cat((mu, s), dim=1)
        if self.keepdim:
            mus.unsqueeze_(dim=self.dim)

        return mus

    def forward_slidwin(self, x, win_length, win_shift, snip_edges=False):
        if isinstance(win_shift, int) and isinstance(win_length, int):
            return self._forward_slidwin_int(x, win_length, win_shift, snip_edges)

        # the window length and/or shift are floats
        return self._forward_slidwin_float(x, win_length, win_shift, snip_edges)

    def _pre_slidwin(self, x, win_length, win_shift, snip_edges):
        if self.dim != -1:
            x = x.transpose(self.dim, -1)

        x, num_frames = self._slidwin_pad(x, win_length, win_shift, snip_edges)
        out_shape = *x.shape[:-1], num_frames
        return x, out_shape

    def _post_slidwin(self, m_x, s_x, out_shape):
        m_x = m_x.view(out_shape)
        s_x = s_x.view(out_shape)
        mus = torch.cat((m_x, s_x), dim=1)
        if self.dim != -1:
            mus = mus.transpose(self.dim, -1).contiguous()

        return mus

    def _forward_slidwin_int(self, x, win_length, win_shift, snip_edges):
        x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)

        c_x = torch.cumsum(x, dim=-1).view(-1, x.shape[-1])
        m_x = (c_x[:, win_shift:] - c_x[:, :-win_shift]) / win_length

        c_x = torch.cumsum(x ** 2, dim=-1).view(-1, x.shape[-1])
        m_x2 = (c_x[:, win_shift:] - c_x[:, :-win_shift]) / win_length
        s_x = torch.sqrt(m_x2 - m_x ** 2).clamp(min=SQRT_EPS)

        mus = self._post_slidwin(m_x, s_x, out_shape)
        return mus

    def _forward_slidwin_float(self, x, win_length, win_shift, snip_edges):

        x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)
        num_frames = out_shape[-1]
        c_x = torch.cumsum(x, dim=-1).view(-1, x.shape[-1])
        c_x2 = torch.cumsum(x ** 2, dim=-1).view(-1, x.shape[-1])

        # xx = x.view(-1, x.shape[-1])
        # print(xx.shape[1])
        # print(torch.max(torch.sum(xx==0, dim=1)))

        m_x = torch.zeros(
            (c_x.shape[0], num_frames), dtype=c_x.dtype, device=c_x.device
        )
        m_x2 = torch.zeros_like(m_x)

        k = 0
        # max_delta = 0
        # max_delta2 = 0
        for i in range(num_frames):
            k1 = int(round(k))
            k2 = int(round(k + win_length))
            m_x[:, i] = (c_x[:, k2] - c_x[:, k1]) / (k2 - k1)
            m_x2[:, i] = (c_x2[:, k2] - c_x2[:, k1]) / (k2 - k1)
            # for j in range(m_x.shape[0]):
            #     m_x_2 = torch.mean(xx[j,k1+1:k2+1])
            #     m_x2_2 = torch.mean(xx[j,k1+1:k2+1]**2)
            #     delta = torch.abs(m_x_2 - m_x[j,i]).item()
            #     delta2 = torch.abs(m_x2_2 - m_x2[j,i]).item()
            #     if (delta > max_delta or delta2 > max_delta2) and (delta>1e-3 or delta2>1e-3):
            #         max_delta = delta
            #         max_delta2 = delta2
            #         print('mx', delta, m_x[j,i], m_x_2)
            #         print('mx2', delta2, m_x2[j,i], m_x2_2)
            #         import sys
            #         sys.stdout.flush()
            #     # if m_x[j,i]**2 > m_x2[j,i]:
            #     #     print('nan')
            #     #     print('mx', m_x[j,i], m_x_2)
            #     #     print('mx2', m_x2[j,i], m_x2_2)
            #     #     print(c_x[j,k2])
            #     #     print(c_x[j,k1])
            #     #     print(c_x2[j,k2])
            #     #     print(c_x2[j,k1])
            #     #     print(xx[j,k1+1:k2+1])
            #     #     raise Exception()

            k += win_shift

        var_x = (m_x2 - m_x ** 2).clamp(min=SQRT_EPS)
        s_x = torch.sqrt(var_x)
        # idx = torch.isnan(s_x) #.any(dim=1)
        # if torch.sum(idx) > 0:
        #     print('sx-nan', s_x[idx])
        #     print('mx-nan', m_x[idx])
        #     print('mx2-nan', m_x2[idx])
        #     print('var-nan', m_x2[idx]-m_x[idx]**2)
        #     #print('cx2-nan', c_x2[idx])
        #     raise Exception()

        mus = self._post_slidwin(m_x, s_x, out_shape)
        return mus

    # def _forward_slidwin_int(self, x, win_length,  win_shift):
    #     num_frames = int((x.shape[self.dim] - win_length + 2*window_shift -1)/win_shift)
    #     pad_right = win_shift * (num_frames - 1) + win_length

    #     if self.dim != -1:
    #         # put pool dim at the end to do the padding
    #         x = x.transpose(self.dim, -1)

    #     xx = nnf.pad(x, (1, pad_right), mode='reflect')
    #     c_x = torch.cumsum(xx, dim=self.dim).transpose(0, -1)

    #     m_x = (c_x[win_shift:] - c_x[:-win_shift]).transpose(0, self.dim)/win_length

    #     c_x = torch.cumsum(xx**2, dim=-1).transpose(0, -1)
    #     m_x2 = (c_x[win_shift:] - c_x[:-win_shift]).transpose(0, self.dim)/win_length
    #     s_x = torch.sqrt(m_x2 - m_x**2).clamp(min=1e-5)
    #     if self.dim == -1:
    #         return torch.cat((m_x, s_x), dim=-2)

    #     return torch.cat((m_x, s_x), dim=-1)

    # def _forward_slidwin_float(self, x, win_shift, win_length):
    #     num_frames = int((x.shape[self.dim] - win_length + 2*window_shift -1)/win_shift)
    #     pad_right = win_shift * (num_frames - 1) + win_length
    #     if self.dim != -1:
    #         x = x.transpose(self.dim, -1)

    #     xx = nnf.pad(x, (1, pad_right), mode='reflect')
    #     c_x = torch.cumsum(xx, dim=-1).transpose(0, -1)
    #     c_x2 = torch.cumsum(xx**2, dim=-1).transpose(0, -1)
    #     m_x = []
    #     m_x2 = []
    #     k = 0
    #     for i in range(num_frames):
    #         k1 = int(math.round(k))
    #         k2 = int(math.round(k+win_length))
    #         w = (k2-k1)
    #         m_x.append((c_x[k2]-c_x[k1])/w)
    #         m_x2.append((c_x2[k2]-c_x2[k1])/w)
    #         k += win_shift

    #     m_x = m_x.cat(tuple(y), dim=0).transpose(0, self.dim).contiguous()
    #     m_x2 = m_x2.cat(tuple(y), dim=0).transpose(0, self.dim).contiguous()
    #     s_x = torch.sqrt(m_x2 - m_x**2).clamp(min=1e-5)
    #     if self.dim == -1:
    #         return torch.cat((m_x, s_x), dim=-2)

    #     return torch.cat((m_x, s_x), dim=-1)


class GlobalMeanLogVarPool1d(_GlobalPool1d):
    """Global mean + log-variance pooling in 1d

    Attributes:
       dim: pooling dimension
       keepdim: it True keeps the same number of dimensions after pooling

    """

    def __init__(self, dim=-1, keepdim=False):
        super().__init__(dim, keepdim)
        self.size_multiplier = 2

    def forward(self, x, weights=None):
        if weights is None:
            mu = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            x2bar = torch.mean(x ** 2, dim=self.dim, keepdim=self.keepdim)
            logvar = torch.log(x2bar - mu * mu + 1e-5)  # for stability in case var=0
            return torch.cat((mu, logvar), dim=-1)

        weights = self._standarize_weights(weights, x.dim())

        xbar = torch.mean(weights * x, dim=self.dim, keepdim=self.keepdim)
        wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        mu = xbar / wbar
        x2bar = torch.mean(weights * x ** 2, dim=self.dim, keepdim=self.keepdim) / wbar
        var = (x2bar - mu * mu).clamp(min=1e-5)
        logvar = torch.log(var)

        return torch.cat((mu, logvar), dim=-1)


class LDEPool1d(_GlobalPool1d):
    """Learnable dictionary encoder pooling in 1d

    Attributes:
       in_feats: input feature dimension
       num_comp: number of cluster components
       dist_pow: power for distance metric
       use_bias: use bias parameter when computing posterior responsibility
       dim: pooling dimension
       keepdim: it True keeps the same number of dimensions after pooling

    """

    def __init__(
        self, in_feats, num_comp=64, dist_pow=2, use_bias=False, dim=-1, keepdim=False
    ):
        super().__init__(dim, keepdim)
        self.mu = nn.Parameter(torch.randn((num_comp, in_feats)))
        self.prec = nn.Parameter(torch.ones((num_comp,)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((num_comp,)))
        else:
            self.bias = 0

        self.dist_pow = dist_pow
        if dist_pow == 1:
            self.dist_f = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dist_f = lambda x: torch.sum(x ** 2, dim=-1)

        self.size_multiplier = num_comp

    @property
    def num_comp(self):
        return self.mu.shape[0]

    @property
    def in_feats(self):
        return self.mu.shape[1]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(in_feats={}, num_comp={}, dist_pow={}, use_bias={}, dim={}, keepdim={})".format(
            self.__class__.__name__,
            self.mu.shape[1],
            self.mu.shape[0],
            self.dist_pow,
            self.use_bias,
            self.dim,
            self.keepdim,
        )
        return s

    def forward(self, x, weights=None):
        if self.dim != 1 or self.dim != -2:
            x = x.transpose(1, self.dim)

        x = torch.unsqueeze(x, dim=2)
        delta = x - self.mu
        dist = self.dist_f(delta)

        llk = -self.prec ** 2 * dist + self.bias
        r = nnf.softmax(llk, dim=-1)
        if weights is not None:
            r *= weights

        r = torch.unsqueeze(r, dim=-1)
        N = torch.sum(r, dim=1) + 1e-9
        F = torch.sum(r * delta, dim=1)
        pool = F / N
        pool = pool.contiguous().view(-1, self.num_comp * self.in_feats)
        if self.keepdim:
            if self.dim == 1 or self.dim == -2:
                pool.unsqueeze_(1)
            else:
                pool.unsqueeze_(-1)

        return pool

    def get_config(self):

        config = {
            "in_feats": self.in_feats,
            "num_comp": self.num_comp,
            "dist_pow": self.dist_pow,
            "use_bias": self.use_bias,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ScaledDotProdAttV1Pool1d(_GlobalPool1d):
    def __init__(
        self, in_feats, num_heads, d_k, d_v, bin_attn=False, dim=-1, keepdim=False
    ):
        super().__init__(dim, keepdim)

        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.bin_attn = bin_attn
        self.q = nn.Parameter(torch.Tensor(1, num_heads, 1, d_k))
        nn.init.orthogonal_(self.q)
        if self.bin_attn:
            self.bias = nn.Parameter(torch.zeros((1, num_heads, 1, 1)))

        self.linear_k = nn.Linear(in_feats, num_heads * d_k)
        self.linear_v = nn.Linear(in_feats, num_heads * d_v)
        self.attn = None
        self.size_multiplier = num_heads * d_v / in_feats

    @property
    def in_feats(self):
        return self.linear_v.in_features

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(in_feats={}, num_heads={}, d_k={}, d_v={}, bin_attn={}, dim={}, keepdim={})".format(
            self.__class__.__name__,
            self.in_feats,
            self.num_heads,
            self.d_k,
            self.d_v,
            self.bin_attn,
            self.dim,
            self.keepdim,
        )
        return s

    def forward(self, x, weights=None):
        batch_size = x.size(0)
        if self.dim != 1:
            x = x.transpose(1, self.dim)

        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_v)
        k = k.transpose(1, 2)  # (batch, head, time, d_k)
        v = v.transpose(1, 2)  # (batch, head, time, d_v)

        scores = torch.matmul(self.q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch, head, 1, time)
        if self.bin_attn:
            scores = torch.sigmoid(scores + self.bias)

        # scores = scores.squeeze(dim=-1)                    # (batch, head, time)
        if weights is not None:
            mask = weights.view(batch_size, 1, 1, -1).eq(0)  # (batch, 1, 1,time)
            if self.bin_attn:
                scores = scores.masked_fill(mask, 0.0)
                self.attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-9)
            else:
                min_value = -1e200
                scores = scores.masked_fill(mask, min_value)
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, head, 1, time)
        else:
            if self.bin_attn:
                self.attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-9)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, 1, time)

        x = torch.matmul(self.attn, v)  # (batch, head, 1, d_v)
        if self.keepdim:
            x = x.view(batch_size, 1, self.num_heads * self.d_v)  # (batch, 1, d_model)
        else:
            x = x.view(batch_size, self.num_heads * self.d_v)  # (batch, d_model)
        return x

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "num_heads": self.num_heads,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "bin_attn": self.bin_attn,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalChWiseAttMeanStdPool1d(_GlobalPool1d):
    """Attentive mean + stddev pooling for each channel"""

    def __init__(
        self,
        in_feats,
        inner_feats=128,
        bin_attn=False,
        use_global_context=True,
        norm_layer=None,
        dim=-1,
        keepdim=False,
    ):
        super().__init__(dim, keepdim)
        self.size_multiplier = 2
        self.in_feats = in_feats
        self.inner_feats = inner_feats
        self.bin_attn = bin_attn

        self.use_global_context = use_global_context
        self.conv1 = _conv1(in_feats, inner_feats)
        if use_global_context:
            self.lin_global = nn.Linear(2 * in_feats, inner_feats, bias=False)
        # torch.autograd.set_detect_anomaly(True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.norm_layer = norm_layer(inner_feats)
        self.activation = nn.Tanh()
        self.conv2 = _conv1(inner_feats, in_feats, bias=True)
        self.stats_pool = GlobalMeanStdPool1d(dim=dim)
        if self.bin_attn:
            self.bias = nn.Parameter(torch.ones((1, in_feats, 1)))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(in_feats={}, inner_feats={}, use_global_context={}, bin_attn={}, dim={}, keepdim={})".format(
            self.__class__.__name__,
            self.in_feats,
            self.inner_feats,
            self.use_global_context,
            self.bin_attn,
            self.dim,
            self.keepdim,
        )
        return s

    def forward(self, x, weights=None):

        x_inner = self.conv1(x)
        # logging.info('x_inner1={} {}'.format(torch.sum(torch.isnan(x_inner)), torch.sum(torch.isinf(x_inner))))
        if self.use_global_context:
            global_mus = self.stats_pool(x)
            x_inner = x_inner + self.lin_global(global_mus).unsqueeze(-1)
        # logging.info('x_inner2={} {}'.format(torch.sum(torch.isnan(x_inner)), torch.sum(torch.isinf(x_inner))))
        attn = self.conv2(self.activation(self.norm_layer(x_inner)))
        if self.bin_attn:
            # attn = torch.sigmoid(attn+self.bias)
            attn = torch.sigmoid(attn)
        else:
            attn = nnf.softmax(attn, dim=-1)

        mus = self.stats_pool(x, weights=attn)
        # logging.info('mus={} {}'.format(torch.sum(torch.isnan(mus)), torch.sum(torch.isinf(mus))))
        return mus

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "inner_feats": self.inner_feats,
            "use_global_context": self.use_global_context,
            "bin_attn": self.bin_attn,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
