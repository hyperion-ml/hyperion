"""
Copyright 2018 Jesus Villalba (Johns Hopkins University)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Tuple, Union

import numpy as np
from jsonargparse import ActionParser, ArgumentParser
from librosa.filters import mel as make_mel_librosa

from ...hyp_defs import float_cpu


class FilterBankFactory:
    """Factory for analysis filter-bank matrices.

    Supported filter-bank types:
      - ``mel_kaldi``: Kaldi-style mel triangular filters.
      - ``mel_etsi``: ETSI-style mel triangular filters.
      - ``mel_librosa``: librosa mel filters (Slaney mel by default).
      - ``mel_librosa_htk``: librosa mel filters in HTK mode.
      - ``linear``: linearly-spaced triangular filters.

    Example:
      >>> from hyperion.np.feats.filter_banks import FilterBankFactory
      >>> fb = FilterBankFactory.create(
      ...     filter_bank_type="mel_kaldi",
      ...     num_filters=23,
      ...     fft_length=512,
      ...     fs=16000,
      ...     low_freq=20,
      ...     high_freq=0,
      ...     norm_filters=True,
      ... )
      >>> fb.shape
      (257, 23)
    """

    @staticmethod
    def _resolve_freq_range(fs: float, low_freq: float, high_freq: float) -> Tuple[float, float]:
        """Resolves and validates low/high cutoff frequencies.

        Args:
          fs: Sample rate in Hz.
          low_freq: Lower cutoff in Hz.
          high_freq: Upper cutoff in Hz. If ``<= 0``, interpreted as offset from
            Nyquist (e.g., ``0`` means Nyquist, ``-100`` means ``Nyquist-100``).

        Returns:
          Validated ``(low_freq, high_freq)`` tuple in Hz.
        """
        if fs <= 0:
            raise ValueError(f"fs must be > 0, got {fs!r}")
        nyquist = fs / 2.0
        if high_freq <= 0:
            high_freq = nyquist + high_freq

        if low_freq < 0:
            raise ValueError(f"low_freq must be >= 0, got {low_freq!r}")
        if high_freq <= low_freq:
            raise ValueError(
                f"high_freq must be greater than low_freq, got low={low_freq!r}, high={high_freq!r}"
            )
        if high_freq > nyquist:
            raise ValueError(
                f"high_freq must be <= Nyquist ({nyquist}), got {high_freq!r}"
            )
        return low_freq, high_freq

    @staticmethod
    def create(
        filter_bank_type: str,
        num_filters: int,
        fft_length: int,
        fs: float,
        low_freq: float,
        high_freq: float,
        norm_filters: bool,
    ) -> np.ndarray:
        """Creates a filter-bank matrix.

        Args:
          filter_bank_type: Filter-bank type.
          num_filters: Number of filters (columns in output matrix).
          fft_length: FFT length used for spectral analysis.
          fs: Sample rate in Hz.
          low_freq: Low cutoff frequency in Hz.
          high_freq: High cutoff frequency in Hz (or offset from Nyquist if ``<= 0``).
          norm_filters: If ``True``, normalizes each filter to unit sum.

        Returns:
          Filter-bank matrix with shape ``(fft_length // 2 + 1, num_filters)``.
        """
        if num_filters < 1:
            raise ValueError(f"num_filters must be >= 1, got {num_filters!r}")
        if fft_length < 2:
            raise ValueError(f"fft_length must be >= 2, got {fft_length!r}")

        if filter_bank_type == "mel_kaldi":
            B = FilterBankFactory.make_mel_kaldi(
                num_filters, fft_length, fs, low_freq, high_freq
            )
        elif filter_bank_type == "mel_etsi":
            B = FilterBankFactory.make_mel_etsi(
                num_filters, fft_length, fs, low_freq, high_freq
            )
        elif filter_bank_type == "mel_librosa":
            B = FilterBankFactory.make_mel_librosa(
                num_filters,
                fft_length,
                fs,
                low_freq,
                high_freq,
                norm_filters=norm_filters,
            )
            norm_filters = False
        elif filter_bank_type == "mel_librosa_htk":
            B = FilterBankFactory.make_mel_librosa(
                num_filters,
                fft_length,
                fs,
                low_freq,
                high_freq,
                htk=True,
                norm_filters=norm_filters,
            )
            norm_filters = False
        elif filter_bank_type == "linear":
            B = FilterBankFactory.make_linear(
                num_filters, fft_length, fs, low_freq, high_freq
            )
        else:
            raise ValueError(f"invalid filter-bank type {filter_bank_type!r}")

        if norm_filters:
            denom = np.sum(B, axis=0, keepdims=True)
            if np.any(denom <= 0):
                raise ValueError(
                    "cannot normalize filters: at least one filter has non-positive sum"
                )
            B = B / denom

        return B

    @staticmethod
    def lin2mel(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts linear frequency (Hz) to mel scale."""
        return 1127.0 * np.log(1 + x / 700)

    @staticmethod
    def mel2lin(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts mel frequency to linear frequency (Hz)."""
        return 700 * (np.exp(x / 1127.0) - 1)

    @staticmethod
    def make_mel_kaldi(
        num_filters: int,
        fft_length: int,
        fs: float,
        low_freq: float,
        high_freq: float,
    ) -> np.ndarray:
        """Builds Kaldi-style mel triangular filters.

        Args:
          num_filters: Number of mel filters.
          fft_length: FFT length.
          fs: Sample rate in Hz.
          low_freq: Low cutoff in Hz.
          high_freq: High cutoff in Hz (or offset from Nyquist if ``<= 0``).

        Returns:
          Filter-bank matrix with shape ``(fft_length // 2 + 1, num_filters)``.
        """
        low_freq, high_freq = FilterBankFactory._resolve_freq_range(
            fs, low_freq, high_freq
        )

        mel_low_freq = FilterBankFactory.lin2mel(low_freq)
        mel_high_freq = FilterBankFactory.lin2mel(high_freq)
        melfc = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
        freqs = np.arange(fft_length // 2 + 1, dtype=float_cpu()) * fs / fft_length
        mels = FilterBankFactory.lin2mel(freqs)

        B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=float_cpu())
        for k in range(num_filters):
            left_mel = melfc[k]
            center_mel = melfc[k + 1]
            right_mel = melfc[k + 2]
            for j in range(int(fft_length / 2 + 1)):
                mel_j = mels[j]
                if mel_j > left_mel and mel_j < right_mel:
                    if mel_j <= center_mel:
                        B[j, k] = (mel_j - left_mel) / (center_mel - left_mel)
                    else:
                        B[j, k] = (right_mel - mel_j) / (right_mel - center_mel)

        return B

    @staticmethod
    def make_mel_etsi(
        num_filters: int,
        fft_length: int,
        fs: float,
        low_freq: float,
        high_freq: float,
    ) -> np.ndarray:
        """Builds ETSI-style mel triangular filters.

        Args:
          num_filters: Number of mel filters.
          fft_length: FFT length.
          fs: Sample rate in Hz.
          low_freq: Low cutoff in Hz.
          high_freq: High cutoff in Hz (or offset from Nyquist if ``<= 0``).

        Returns:
          Filter-bank matrix with shape ``(fft_length // 2 + 1, num_filters)``.
        """
        low_freq, high_freq = FilterBankFactory._resolve_freq_range(
            fs, low_freq, high_freq
        )
        mel_low_freq = FilterBankFactory.lin2mel(low_freq)
        mel_high_freq = FilterBankFactory.lin2mel(high_freq)
        fc = FilterBankFactory.mel2lin(
            np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
        )
        cbin = np.round(fc / fs * fft_length).astype(int)

        B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=float_cpu())
        for k in range(num_filters):
            for j in range(cbin[k], cbin[k + 1] + 1):
                B[j, k] = (j - cbin[k] + 1) / (cbin[k + 1] - cbin[k] + 1)
            for j in range(cbin[k + 1] + 1, cbin[k + 2] + 1):
                B[j, k] = (cbin[k + 2] - j + 1) / (cbin[k + 2] - cbin[k + 1] + 1)

        return B

    @staticmethod
    def make_linear(
        num_filters: int,
        fft_length: int,
        fs: float,
        low_freq: float,
        high_freq: float,
    ) -> np.ndarray:
        """Builds linearly-spaced triangular filters.

        Args:
          num_filters: Number of linear filters.
          fft_length: FFT length.
          fs: Sample rate in Hz.
          low_freq: Low cutoff in Hz.
          high_freq: High cutoff in Hz (or offset from Nyquist if ``<= 0``).

        Returns:
          Filter-bank matrix with shape ``(fft_length // 2 + 1, num_filters)``.
        """
        low_freq, high_freq = FilterBankFactory._resolve_freq_range(
            fs, low_freq, high_freq
        )
        fc = np.linspace(low_freq, high_freq, num_filters + 2)
        cbin = np.round(fc / fs * fft_length).astype(int)

        B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=float_cpu())
        for k in range(num_filters):
            for j in range(cbin[k], cbin[k + 1] + 1):
                B[j, k] = (j - cbin[k] + 1) / (cbin[k + 1] - cbin[k] + 1)
            for j in range(cbin[k + 1] + 1, cbin[k + 2] + 1):
                B[j, k] = (cbin[k + 2] - j + 1) / (cbin[k + 2] - cbin[k + 1] + 1)

        return B

    @staticmethod
    def make_mel_librosa(
        num_filters: int,
        fft_length: int,
        fs: float,
        low_freq: float,
        high_freq: float,
        htk: bool = False,
        norm_filters: bool = False,
    ) -> np.ndarray:
        """Builds mel filter banks using librosa.

        Args:
          num_filters: Number of mel filters.
          fft_length: FFT length.
          fs: Sample rate in Hz.
          low_freq: Low cutoff in Hz.
          high_freq: High cutoff in Hz (or offset from Nyquist if ``<= 0``).
          htk: If ``True``, uses HTK mel conversion in librosa.
          norm_filters: If ``True``, applies Slaney-style area normalization in librosa.

        Returns:
          Filter-bank matrix with shape ``(fft_length // 2 + 1, num_filters)``.
        """
        low_freq, high_freq = FilterBankFactory._resolve_freq_range(
            fs, low_freq, high_freq
        )

        if norm_filters:
            norm = "slaney"
        else:
            norm = None

        return make_mel_librosa(
            fs,
            fft_length,
            num_filters,
            fmin=low_freq,
            fmax=high_freq,
            htk=htk,
            norm=norm,
        ).T

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds filter-bank factory options to a parser.

        Args:
          parser: Argument parser.
          prefix: Optional argument namespace prefix.

        Returns:
          ``None``.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

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
            help=(
                "Filter-bank type: mel_kaldi, mel_etsi, mel_librosa, "
                "mel_librosa_htk, linear"
            ),
        )

        parser.add_argument(
            "--num-filters",
            type=int,
            default=23,
            help="Number of triangular mel-frequency bins",
        )

        parser.add_argument(
            "--low-freq",
            type=float,
            default=20,
            help="Low cutoff frequency for mel bins",
        )

        parser.add_argument(
            "--high-freq",
            type=float,
            default=0,
            help="High cutoff frequency for mel bins (if < 0, offset from Nyquist)",
        )

        parser.add_argument(
            "--norm-filters",
            default=False,
            action="store_true",
            help="Normalize filters coeff to sum up to 1",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
