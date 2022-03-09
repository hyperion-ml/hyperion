"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn
import torch.nn.functional as nnf

count = 0


class AxisMasker(nn.Module):
    """Applies a mask to the spectrogram along time or freq dimension.
    Implementation based on espnet.

    Attributes:
      min_width: minimum width of the mask.
      max_width: maximum width of the mask.
      min_num_mask: minimum number of masks.
      max_num_mask: maximum number of masks.
      dim: axis where we apply the mask
      fill_value: masking value
    """

    def __init__(
        self,
        min_width=0,
        max_width=30,
        min_num_masks=1,
        max_num_masks=2,
        dim=-1,
        fill_value=0,
    ):
        super().__init__()
        assert min_width >= 0
        assert max_width > 0
        assert min_num_masks >= 0
        assert max_num_masks > 0

        self.min_width = min_width
        self.max_width = max_width
        self.min_num_masks = min_num_masks
        self.max_num_masks = max_num_masks
        self.dim = dim
        self.fill_value = fill_value

    def __repr__(self):
        s = (
            "{}(min_width={}, max_width={}, "
            "min_num_masks={}, max_num_masks={}, "
            "dim={}, fill_value={})"
        ).format(
            self.__class__.__name__,
            self.min_width,
            self.max_width,
            self.min_num_masks,
            self.max_num_masks,
            self.dim,
            self.fill_value,
        )
        return s

    def forward(self, x):
        """Apply mask along time or freq dimension

        Args:
           x: spectrogram (batch, *, time, freq)

        Returns:
           Masked spectrogram (batch, *, time, freq)
        """
        if not self.training:
            return x

        in_shape = x.shape
        ndim = x.dim()
        if ndim > 3:
            x = x.view(-1, x.shape[-2], x.shape[-1])

        batch_size = x.shape[0]
        masked_dim_length = x.shape[self.dim]
        # select how many masks
        num_masks = torch.randint(
            self.min_num_masks, self.max_num_masks + 1, size=(1,), device=x.device
        )[0]
        # (batch, num_mask, 1)
        widths = torch.randint(
            self.min_width,
            self.max_width + 1,
            size=(batch_size, num_masks),
            device=x.device,
        ).unsqueeze(-1)

        max_start_pos = masked_dim_length - torch.max(widths) + 1
        # (batch, num_mask, 1)
        start_pos = torch.randint(
            0, max_start_pos, size=(batch_size, num_masks), device=x.device
        ).unsqueeze(-1)
        # (1, 1, masked_dim_length)
        ref = torch.arange(masked_dim_length, device=x.device).view(1, 1, -1)
        # (batch, num_mask, mask_dim_length)
        mask = (start_pos <= ref) * (ref < (start_pos + widths))
        # (batch, mask_dim_length)
        mask = mask.any(dim=1)  # multiply all masks

        if self.dim == -1 or self.dim == ndim - 1:
            mask = mask.unsqueeze(-2)
        else:
            mask = mask.unsqueeze(-1)

        x = x.masked_fill(mask, self.fill_value)
        if ndim > 3:
            x = x.view(in_shape)

        return x


class SpecWarper(nn.Module):
    """Warps the spectrogram along time or freq dimension.
    Implementation based on espnet.

    Attributes:
      window: time warp parameter.
      mode: interpolation mode in ["nearest", "linear", "bilinear"]
      dim: warping dimension.
    """

    def __init__(self, window=80, mode="bicubic", dim=-2):
        super().__init__()
        self.window = window
        self.mode = mode
        self.dim = dim

    def __repr__(self):
        s = ("{}(window={}, mode={}, dim={}").format(
            self.__class__.__name__, self.window, self.mode, self.dim
        )
        return s

    def forward(self, x, x_lengths=None):
        """warps x along time or freq dimension

        Args:
           x: spectrogram shape= (batch, *, time, freq)
           lengths: time lengths of the sequences.
        Returns:
           warped spectrogram shape = (batch, *, time, freq)
        """
        if not self.training:
            return x

        in_shape = x.shape
        ndim = x.dim()
        if ndim == 3:
            x = x.unsqueeze(1)

        if self.dim > 0:
            dim = ndim - self.dim
        else:
            dim = self.dim

        # for warping in freq dimension
        if dim == -1:
            x = x.transpose(-1, -2)

        # to make it batcheable we are going to warp
        # the first n frames where n is the length of the
        # shortest utterance
        # the end of the utterance will not be warped
        if dim == -1 or x_lengths is None:
            warp_length = x.shape[-2]
        else:
            warp_length = int(x.shape[-2] * torch.min(x_lengths))

        center = torch.randint(self.window, warp_length - self.window, (1,))[0]
        warped = torch.randint(center - self.window, center + self.window, (1,))[0] + 1

        # (batch, C, warped, freq)
        left = nnf.interpolate(
            x[:, :, :center], (warped, x.shape[3]), mode=self.mode, align_corners=False
        )
        # (batch, C, time - warped, Freq)
        right = torch.nn.functional.interpolate(
            x[:, :, center:warp_length],
            (warp_length - warped, x.shape[3]),
            mode=self.mode,
            align_corners=False,
        )

        if warp_length != x.shape[-2]:
            right_nowarp = x[:, :, warp_length:]
            x = torch.cat([left, right, right_nowarp], dim=-2)
        else:
            x = torch.cat([left, right], dim=-2)

        if dim == -1:
            x = x.transpose(-1, -2)

        x = x.view(in_shape)
        return x


class SpecAugment(nn.Module):
    """Implementation of SpecAugment.

    Reference:
     Daniel S. Park et al.
     "SpecAugment: A Simple Data
      Augmentation Method for Automatic Speech Recognition"

    Attributes:
      time_warp_prob:   probability of applying time warping.
      time_warp_window: time warp parameter.
      time_warp_mode:   interpolation mode in ["nearest", "linear", "bilinear"]
      time_mask_prob:   probability of applying masking in time.
      time_min_width:   minimum width of the time mask.
      time_max_width:   maximum width of the time mask.
      time_min_num_mask: minimum number of time masks.
      time_max_num_mask: maximum number of time masks.
      freq_mask_prob:    probability of applying frequency masking.
      freq_min_width:    minimum width of the frequency mask.
      freq_max_width:    maximum width of the frequency mask.
      freq_min_num_mask: minimum number of frequency masks.
      freq_max_num_mask: maximum number of frequency masks.
      fill_value:        masking value.
    """

    def __init__(
        self,
        time_warp_prob=0,
        time_warp_window=5,
        time_warp_mode="bicubic",
        time_mask_prob=0,
        time_mask_min_width=0,
        time_mask_max_width=100,
        time_mask_min_num_masks=1,
        time_mask_max_num_masks=2,
        freq_mask_prob=0,
        freq_mask_min_width=0,
        freq_mask_max_width=20,
        freq_mask_min_num_masks=1,
        freq_mask_max_num_masks=2,
        fill_value=0,
    ):

        super().__init__()
        self.time_warp_prob = time_warp_prob
        self.time_warp_window = time_warp_window
        self.time_warp_mode = time_warp_mode
        self.time_mask_prob = time_mask_prob
        self.time_mask_min_width = time_mask_min_width
        self.time_mask_max_width = time_mask_max_width
        self.time_mask_min_num_masks = time_mask_min_num_masks
        self.time_mask_max_num_masks = time_mask_max_num_masks
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_min_width = freq_mask_min_width
        self.freq_mask_max_width = freq_mask_max_width
        self.freq_mask_min_num_masks = freq_mask_min_num_masks
        self.freq_mask_max_num_masks = freq_mask_max_num_masks
        self.fill_value = fill_value

        self.time_masker = None
        self.freq_masker = None
        self.time_warper = None

        if self.time_mask_prob > 0:
            self.time_masker = AxisMasker(
                min_width=time_mask_min_width,
                max_width=time_mask_max_width,
                min_num_masks=time_mask_min_num_masks,
                max_num_masks=time_mask_max_num_masks,
                dim=-2,
                fill_value=fill_value,
            )

        if self.freq_mask_prob > 0:
            self.freq_masker = AxisMasker(
                min_width=freq_mask_min_width,
                max_width=freq_mask_max_width,
                min_num_masks=freq_mask_min_num_masks,
                max_num_masks=freq_mask_max_num_masks,
                dim=-1,
                fill_value=fill_value,
            )

        if self.time_warp_prob > 0:
            self.time_warper = SpecWarper(
                window=time_warp_window, mode=time_warp_mode, dim=-2
            )

    def __repr__(self):
        s = (
            "{}(time_warper(p={})={}, time_masker(p={})={}, freq_masker(p={})={})"
        ).format(
            self.__class__.__name__,
            self.time_warp_prob,
            self.time_warper,
            self.time_mask_prob,
            self.time_masker,
            self.freq_mask_prob,
            self.freq_masker,
        )
        return s

    def forward(self, x, x_lengths=None):
        """Applies spec augment to input
        Args:
           x: spectrogram with shape = (batch, time, freq)
           lengths: time lengths of the sequences.
        Returns:
           Augmented spectrogram with shape = (batch, time, freq)
        """
        if not self.training:
            return x
        # global count
        # import matplotlib
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.tight_layout()
        # ax = plt.subplot(221)
        # ax.imshow(x.cpu().numpy()[0].T)
        r = torch.rand((3,), device=x.device)
        if self.time_warp_prob > r[0]:
            x = self.time_warper(x, x_lengths)
            # ax = plt.subplot(222)
            # ax.imshow(x.cpu().numpy()[0].T)

        if self.time_mask_prob > r[1]:
            x = self.time_masker(x)
            # ax = plt.subplot(223)
            # ax.imshow(x.cpu().numpy()[0].T)

        if self.freq_mask_prob > r[2]:
            x = self.freq_masker(x)
            # ax = plt.subplot(224)
            # ax.imshow(x.cpu().numpy()[0].T)

        # plt.savefig("spec_aug%d.png" % count, dpi=600)
        # plt.close()
        # count += 1
        return x

    @staticmethod
    def filter_args(**kwargs):
        """Filters SpecAugment args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with SpecAugment options.
        """
        valid_args = (
            "time_warp_prob",
            "time_warp_window",
            "time_warp_mode",
            "time_mask_prob",
            "time_mask_max_width",
            "time_mask_min_width",
            "time_mask_max_num_masks",
            "time_mask_min_num_masks",
            "freq_mask_prob",
            "freq_mask_max_width",
            "freq_mask_min_width",
            "freq_mask_max_num_masks",
            "freq_mask_min_num_masks",
            "fill_value",
        )

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds SpecAugment options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--time-warp-prob",
            type=float,
            default=0.0,
            help="prob. for applying warping",
        )
        parser.add_argument(
            "--time-warp-window", type=int, default=80, help="time warp window param."
        )
        parser.add_argument(
            "--time-warp-mode",
            default="bicubic",
            choices=["bilinear", "linear", "nearest", "bicubic", "trilinear"],
            help="prob. for applying warping",
        )

        parser.add_argument(
            "--time-mask-prob",
            type=float,
            default=0.0,
            help="prob. for applying time masking",
        )
        parser.add_argument(
            "--time-mask-min-width",
            type=int,
            default=0,
            help="min. width for time mask",
        )
        parser.add_argument(
            "--time-mask-max-width",
            type=int,
            default=100,
            help="max. width for time mask",
        )
        parser.add_argument(
            "--time-mask-min-num-masks",
            type=int,
            default=1,
            help="min. number of time mask",
        )
        parser.add_argument(
            "--time-mask-max-num-masks",
            type=int,
            default=2,
            help="max. number of time mask",
        )

        parser.add_argument(
            "--freq-mask-prob",
            type=float,
            default=0.0,
            help="prob. for applying freq. masking",
        )
        parser.add_argument(
            "--freq-mask-min-width",
            type=int,
            default=0,
            help="min. width for freq mask",
        )
        parser.add_argument(
            "--freq-mask-max-width",
            type=int,
            default=100,
            help="max. width for freq mask",
        )
        parser.add_argument(
            "--freq-mask-min-num-masks",
            type=int,
            default=1,
            help="min. number of freq mask",
        )
        parser.add_argument(
            "--freq-mask-max-num-masks",
            type=int,
            default=2,
            help="max. number of freq mask",
        )

        parser.add_argument(
            "--fill-value",
            type=float,
            default=0.0,
            help="filling value for the masked spec. bins",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
