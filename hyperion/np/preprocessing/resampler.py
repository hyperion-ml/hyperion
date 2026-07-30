"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torchaudio.transforms as tat


class ResamplerToTargetFreq:
    """Caches and applies resamplers to a fixed target sample frequency.

    Attributes:
      target_sample_freq: Target sampling rate in Hz.
      resamplers: Cached mapping from input sampling rate to callable resampler.

    Example:
      ```python
      import numpy as np
      from hyperion.np.preprocessing.resampler import ResamplerToTargetFreq

      x = np.random.randn(32000).astype(np.float32)  # 2s at 16kHz
      rs = ResamplerToTargetFreq(8000)
      y, fs = rs(x, input_sample_freq=16000)
      # y is resampled to fs=8000
      ```
    """

    def __init__(self, target_sample_freq: float):
        """Initializes the resampler cache.

        Args:
          target_sample_freq: Target sampling rate in Hz.
        """
        if target_sample_freq <= 0:
            raise ValueError(
                f"target_sample_freq must be > 0, got {target_sample_freq!r}"
            )
        self.target_sample_freq = target_sample_freq
        self.resamplers: Dict[float, Callable[[np.ndarray], np.ndarray]] = {}

    def get_resampler(
        self, input_sample_freq: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Returns a cached resampling function for an input sample rate.

        Args:
          input_sample_freq: Input sampling rate in Hz.

        Returns:
          Callable that maps input waveform arrays to resampled arrays.
        """
        if input_sample_freq <= 0:
            raise ValueError(
                f"input_sample_freq must be > 0, got {input_sample_freq!r}"
            )
        if input_sample_freq in self.resamplers:
            return self.resamplers[input_sample_freq]

        try:
            resampler = tat.Resample(
                int(input_sample_freq),
                int(self.target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        except Exception:
            resampler = tat.Resample(
                int(input_sample_freq),
                int(self.target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        resampler_f = (
            lambda x: resampler(torch.from_numpy(x).float())
            .numpy()
            .astype(x.dtype, copy=False)
        )
        self.resamplers[input_sample_freq] = resampler_f
        return resampler_f

    def __call__(
        self, x: np.ndarray, input_sample_freq: float
    ) -> Tuple[np.ndarray, float]:
        """Resamples a waveform to the target sampling frequency.

        Args:
          x: Input waveform array.
          input_sample_freq: Input sampling rate in Hz.

        Returns:
          Tuple with the resampled waveform and the target sample rate.
        """
        if input_sample_freq == self.target_sample_freq:
            return x, input_sample_freq

        resampler = self.get_resampler(input_sample_freq)
        return resampler(x), self.target_sample_freq


class ResamplerFromInputFreq:
    """Caches and applies resamplers from a fixed input sample frequency.

    Attributes:
      input_sample_freq: Input sampling rate in Hz.
      resamplers: Cached mapping from target sampling rate to callable resampler.

    Example:
      ```python
      import numpy as np
      from hyperion.np.preprocessing.resampler import ResamplerFromInputFreq

      x = np.random.randn(16000).astype(np.float32)  # 1s at 16kHz
      rs = ResamplerFromInputFreq(16000)
      y, fs = rs(x, target_sample_freq=22050)
      # y is resampled to fs=22050
      ```
    """

    def __init__(self, input_sample_freq: float):
        """Initializes the resampler cache.

        Args:
          input_sample_freq: Input sampling rate in Hz.
        """
        if input_sample_freq <= 0:
            raise ValueError(
                f"input_sample_freq must be > 0, got {input_sample_freq!r}"
            )
        self.input_sample_freq = input_sample_freq
        self.resamplers: Dict[float, Callable[[np.ndarray], np.ndarray]] = {}

    def get_resampler(
        self, target_sample_freq: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Returns a cached resampling function for a target sample rate.

        Args:
          target_sample_freq: Target sampling rate in Hz.

        Returns:
          Callable that maps input waveform arrays to resampled arrays.
        """
        if target_sample_freq <= 0:
            raise ValueError(
                f"target_sample_freq must be > 0, got {target_sample_freq!r}"
            )
        if target_sample_freq in self.resamplers:
            return self.resamplers[target_sample_freq]

        try:
            resampler = tat.Resample(
                int(self.input_sample_freq),
                int(target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        except Exception:
            resampler = tat.Resample(
                int(self.input_sample_freq),
                int(target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        resampler_f = (
            lambda x: resampler(torch.from_numpy(x).float())
            .numpy()
            .astype(x.dtype, copy=False)
        )
        self.resamplers[target_sample_freq] = resampler_f
        return resampler_f

    def __call__(
        self, x: np.ndarray, target_sample_freq: float
    ) -> Tuple[np.ndarray, float]:
        """Resamples a waveform from the configured input sample rate.

        Args:
          x: Input waveform array.
          target_sample_freq: Target sampling rate in Hz.

        Returns:
          Tuple with the resampled waveform and target sample rate.
        """
        if self.input_sample_freq == target_sample_freq:
            return x, target_sample_freq

        resampler = self.get_resampler(target_sample_freq)
        return resampler(x), target_sample_freq


class Any2AnyFreqResampler:
    """Caches and applies resamplers between arbitrary input/target rates.

    Attributes:
      resamplers: Cached mapping from `(input_freq, target_freq)` to callable resampler.

    Example:
      ```python
      import numpy as np
      from hyperion.np.preprocessing.resampler import Any2AnyFreqResampler

      x = np.random.randn(24000).astype(np.float32)  # 1.5s at 16kHz
      rs = Any2AnyFreqResampler()
      y, fs = rs(x, input_sample_freq=16000, target_sample_freq=8000)
      # y is resampled to fs=8000
      ```
    """

    def __init__(self):
        """Initializes the resampler cache."""
        self.resamplers: Dict[
            Tuple[float, float], Callable[[np.ndarray], np.ndarray]
        ] = {}

    def get_resampler(
        self, input_sample_freq: float, target_sample_freq: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Returns a cached resampling function for `(input_freq, target_freq)`.

        Args:
          input_sample_freq: Input sampling rate in Hz.
          target_sample_freq: Target sampling rate in Hz.

        Returns:
          Callable that maps input waveform arrays to resampled arrays.
        """
        if input_sample_freq <= 0:
            raise ValueError(
                f"input_sample_freq must be > 0, got {input_sample_freq!r}"
            )
        if target_sample_freq <= 0:
            raise ValueError(
                f"target_sample_freq must be > 0, got {target_sample_freq!r}"
            )

        key = (input_sample_freq, target_sample_freq)
        if key in self.resamplers:
            return self.resamplers[key]

        try:
            resampler = tat.Resample(
                int(input_sample_freq),
                int(target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        except Exception:
            resampler = tat.Resample(
                int(input_sample_freq),
                int(target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        resampler_f = (
            lambda x: resampler(torch.from_numpy(x).float())
            .numpy()
            .astype(x.dtype, copy=False)
        )
        self.resamplers[key] = resampler_f
        return resampler_f

    def __call__(
        self, x: np.ndarray, input_sample_freq: float, target_sample_freq: float
    ) -> Tuple[np.ndarray, float]:
        """Resamples a waveform between arbitrary sampling rates.

        Args:
          x: Input waveform array.
          input_sample_freq: Input sampling rate in Hz.
          target_sample_freq: Target sampling rate in Hz.

        Returns:
          Tuple with the resampled waveform and target sample rate.
        """
        if input_sample_freq == target_sample_freq:
            return x, input_sample_freq

        resampler = self.get_resampler(input_sample_freq, target_sample_freq)
        dtype = x.dtype
        x = resampler(x.astype(np.float32, copy=False)).astype(dtype, copy=False)
        return x, target_sample_freq
