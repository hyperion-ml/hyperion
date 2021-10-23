"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import time
import logging
import math

import torch
import torch.nn as nn

from ...torch_model import TorchModel
from ...utils.misc import l2_norm, get_selfsim_tarnon


class PLDABase(TorchModel):
    def __init__(
        self,
        x_dim=None,
        mu=None,
        num_classes=0,
        x_ref=None,
        p_tar=0.05,
        margin_multi=0.3,
        margin_tar=0.3,
        margin_non=0.3,
        margin_warmup_epochs=10,
        adapt_margin=False,
        adapt_gamma=0.99,
        lnorm=False,
        var_floor=1e-5,
        prec_floor=1e-5,
        preprocessor=None,
    ):
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
    def l2_norm(x):
        return math.sqrt(x.shape[-1]) * l2_norm(x)

    def __repr__(self):
        return self.__str__()

    def update_margin(self, epoch):
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

    def _adapt_margin_multi(self, llr, llr_tar):
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

    def _adapt_margin_bin(self, llr, y_tar, y_non):
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

    def _apply_margin_multi(self, llr, y=None):
        if y is None or not self.training or self.cur_margin_multi == 0:
            return llr

        batch_size = len(llr)
        idx_ = torch.arange(0, batch_size, dtype=torch.long)
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

    def _apply_margin_bin(self, llr, y=None, y_bin=None):
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

        y_non = 1 - y_bin
        if self.adapt_margin:
            y_tar = y_bin - torch.eye(len(y), dtype=torch.get_default_dtype())
            self._adapt_margin_bin(llr, y_tar, y_non)
            del y_tar
            margin_tar = self.cur_margin_tar * self.max_margin_tar
            margin_non = self.cur_margin_non * self.max_margin_non
        else:
            margin_tar = self.cur_margin_tar
            margin_non = self.cur_margin_non

        llr_m = y_bin * (llr - margin_tar) + y_non * (llr + margin_non)
        return llr_m

    def forward(self, x, y=None, return_multi=True, return_bin=True, y_bin=None):
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
    def compute_stats_hard(x, y, order=2, sample_weight=None, scale_factor=1):
        x_dim = x.shape[1]
        num_classes = torch.max(y) + 1
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

    def llr_Nvs1(self, x1, x2, y1=None, method="vavg", preproc=True):
        if self.preprocessor is not None and preproc:
            x1 = self.preprocessor(x1)
            x2 = self.preprocessor(x2)

        if y1 is not None:
            N1, F1 = self.compute_stats_hard(x1, y1)
            x1 = F1 / N1.unsqueeze(-1)

        return self.llr_1vs1(x1, x2, preproc=False)

    def get_config(self):
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
