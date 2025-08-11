"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..layers.audio_feats import Wav2LogFilterBank
from .convergence_loss import ConvergenceLoss


class MultiResolutionFilterBankLoss(nn.Module):
    """
    Computes a multi-resolution filter bank loss between two waveforms.

    This loss computes filterbank features at multiple resolutions (e.g., different
    frame lengths and shifts) and compares the log-magnitude filterbanks using L1 loss.
    Optionally, a convergence loss (relative Frobenius norm) can be added to improve
    convergence behavior.

    This is useful in speech and audio modeling tasks where spectral resolution
    at different temporal scales matters.

    Args:
        sample_frequency (int): Sample rate of input waveforms (e.g., 16000 Hz).
        frame_lengths (List[float]): List of frame lengths (in ms) for each resolution.
        frame_shifts (Optional[List[float]]): List of frame shifts (in ms). If None, defaults to frame_length // 4.
        remove_dc_offset (bool): Whether to remove DC offset from waveform before processing.
        window_type (str): Type of window function to use. One of:
            ["hamming", "hanning", "povey", "rectangular", "blackman"].
        use_fft_mag (bool): If True, use |X(f)|; if False, use |X(f)|².
        dither (float): Amount of dithering to add to waveform.
        fb_type (str): Filter bank type. One of:
            ["mel_kaldi", "mel_etsi", "mel_librosa", "mel_librosa_htk", "linear"].
        low_freqs (Union[float, List[float]]): Low cutoff frequency or list per resolution.
        high_freqs (Union[float, List[float]]): High cutoff frequency or list per resolution.
            If 0 or negative, it's interpreted relative to Nyquist.
        num_filters (Union[int, List[int]]): Number of filters (per resolution).
        norm_filters (bool): Whether to normalize filter coefficients.
        snip_edges (bool): Whether to snip frames at edges or reflect data.
        center (bool): If True, centers window at t*frame_shift. Overrides snip_edges.
        reduction (str): Reduction method for the loss ('mean', 'sum', or 'none').
        compute_conv_loss (bool): Whether to compute convergence loss alongside L1 loss.
    """

    def __init__(
        self,
        sample_frequency: int = 16000,
        frame_lengths: List[float] = [25.0],
        frame_shifts: Optional[List[float]] = None,
        remove_dc_offset: bool = True,
        window_type: str = "povey",
        use_fft_mag: bool = False,
        dither: float = 1.0 / 2**15,
        fb_type: str = "mel_kaldi",
        low_freqs: Union[float, List[float]] = [20.0],
        high_freqs: Union[float, List[float]] = [0.0],
        num_filters: Union[int, List[int]] = [80],
        norm_filters: bool = False,
        snip_edges: bool = True,
        center: bool = False,
        reduction: str = "mean",
        compute_conv_loss: bool = True,
    ):
        super().__init__()
        num_resolutions = len(frame_lengths)
        if frame_shifts is None:
            frame_shifts = [int(f // 4) for f in frame_lengths]

        if isinstance(low_freqs, float):
            low_freqs = [low_freqs] * num_resolutions
        elif len(low_freqs) == 1:
            low_freqs = low_freqs * num_resolutions
        else:
            assert (
                len(low_freqs) == num_resolutions
            ), "low_freqs must be a single value or a list of length num_resolutions"

        if isinstance(high_freqs, float):
            high_freqs = [high_freqs] * num_resolutions
        elif len(high_freqs) == 1:
            high_freqs = high_freqs * num_resolutions
        else:
            assert (
                len(high_freqs) == num_resolutions
            ), "high_freqs must be a single value or a list of length num_resolutions"

        if isinstance(num_filters, int):
            num_filters = [num_filters] * num_resolutions
        elif len(num_filters) == 1:
            num_filters = num_filters * num_resolutions
        else:
            assert (
                len(num_filters) == num_resolutions
            ), "num_filters must be a single value or a list of length num_resolutions"

        log_fb_extractors = []
        for i in range(num_resolutions):
            fft_length = 2 ** (
                (int(frame_lengths[i] * sample_frequency / 1000) - 1).bit_length()
            )
            extractor = Wav2LogFilterBank(
                fs=sample_frequency,
                frame_length=frame_lengths[i],
                frame_shift=frame_shifts[i],
                fft_length=fft_length,
                remove_dc_offset=remove_dc_offset,
                window_type=window_type,
                use_fft_mag=use_fft_mag,
                dither=dither,
                fb_type=fb_type,
                low_freq=low_freqs[i],
                high_freq=high_freqs[i],
                num_filters=num_filters[i],
                norm_filters=norm_filters,
                snip_edges=snip_edges,
                center=center,
                preemph_coeff=0.0,  # Pre-emphasis is not used in filter bank loss
            )
            log_fb_extractors.append(extractor)

        self.log_fb_extractors = nn.ModuleList(log_fb_extractors)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        if compute_conv_loss:
            self.conv_loss = ConvergenceLoss(reduction=reduction)

        self.compute_conv_loss = compute_conv_loss

    def forward(
        self, x_pred: torch.Tensor, x_ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes multi-resolution mel filter-bank loss between an estimate and a reference
        signal.
        Args:
            x_pred (torch.Tensor): Estimated audio signal of shape (B, T).
            x_ref (torch.Tensor): Reference audio signal of shape (B, T).
        Returns:
            torch.Tensor: Computed multi-resolution STFT loss.
        """
        loss_sc = 0.0
        loss_mag = 0.0
        for extractor in self.log_fb_extractors:
            X_pred = extractor(x_pred)
            X_ref = extractor(x_ref)
            loss_mag += self.l1_loss(
                X_pred,
                X_ref,
            )
            if self.compute_conv_loss:
                loss_sc += self.conv_loss(X_pred.exp(), X_ref.exp())

        return loss_sc, loss_mag

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds feature extractor options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--sample-frequency",
            default=16000,
            type=int,
            help=(
                "Waveform data sample frequency (must match the waveform file, "
                "if specified there)"
            ),
        )

        parser.add_argument(
            "--frame-lengths",
            type=float,
            default=[2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            nargs="+",
            help="Frame length in milliseconds",
        )
        parser.add_argument(
            "--frame-shifts",
            type=float,
            default=None,
            nargs="+",
            help="Frame shift in milliseconds",
        )
        parser.add_argument(
            "--remove-dc-offset",
            default=True,
            action=ActionYesNo,
            help="If true, it removes the DC offset from the waveform",
        )

        parser.add_argument(
            "--window-type",
            default="povey",
            choices=["hamming", "hanning", "povey", "rectangular", "blackman"],
            help=(
                'Type of window ("hamming"|"hanning"|"povey"|'
                '"rectangular"|"blackman")'
            ),
        )

        parser.add_argument(
            "--use-fft-mag",
            default=False,
            action=ActionYesNo,
            help="If true, it uses |X(f)|, if false, it uses |X(f)|^2",
        )

        parser.add_argument(
            "--dither",
            type=float,
            default=1.0 / 2**15,
            help="Dithering constant (0.0 means no dither)",
        )

        parser.add_argument(
            "--fb-type",
            default="mel_kaldi",
            choices=[
                "mel_kaldi",
                "mel_etsi",
                "mel_librosa",
                "mel_librosa_htk",
                "linear",
            ],
            help="Filter-bank type: mel_kaldi, mel_etsi, linear",
        )

        parser.add_argument(
            "--num-filters",
            type=int,
            default=[5, 10, 20, 40, 80, 160, 320],
            nargs="+",
            help="Number of triangular mel-frequency bins",
        )

        parser.add_argument(
            "--low-freqs",
            type=float,
            default=[20],
            nargs="+",
            help="Low cutoff frequency for mel bins",
        )

        parser.add_argument(
            "--high-freqs",
            type=float,
            default=[0],
            nargs="+",
            help="High cutoff frequency for mel bins (if < 0, offset from Nyquist)",
        )

        parser.add_argument(
            "--norm-filters",
            default=False,
            action=ActionYesNo,
            help="Normalize filters coeff to sum up to 1",
        )

        parser.add_argument(
            "--snip-edges",
            default=True,
            action=ActionYesNo,
            help=(
                "If true, end effects will be handled by outputting only "
                "frames that completely fit in the file, and the number of "
                "frames depends on the frame-length.  If false, the number "
                "of frames depends only on the frame-shift, "
                "and we reflect the data at the ends."
            ),
        )

        parser.add_argument(
            "--center",
            default=False,
            action=ActionYesNo,
            help=(
                "If true, puts the center of the frame at t*frame_shift, "
                "it over-wrides snip-edges and set it to false"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
