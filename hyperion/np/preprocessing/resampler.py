"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import torch
import torchaudio.transforms as tat


class ResamplerToTargetFreq:
    def __init__(self, target_sample_freq: float):
        self.target_sample_freq = target_sample_freq
        self.resamplers = {}

    def get_resampler(self, input_sample_freq):
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
        except:
            resampler = tat.Resample(
                int(input_sample_freq),
                int(self.target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        resampler_f = lambda x: resampler(torch.from_numpy(x)).numpy()
        self.resamplers[input_sample_freq] = resampler_f
        return resampler_f

    def __call__(self, x, input_sample_freq: float):
        if input_sample_freq == self.target_sample_freq:
            return x, input_sample_freq

        resampler = self.get_resampler(input_sample_freq)
        return resampler(x), self.target_sample_freq


class ResamplerFromInputFreq:
    def __init__(self, input_sample_freq: float):
        self.input_sample_freq = input_sample_freq
        self.resamplers = {}

    def get_resampler(self, target_sample_freq):
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
        except:
            resampler = tat.Resample(
                int(self.input_sample_freq),
                int(target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        resampler_f = lambda x: resampler(torch.from_numpy(x)).numpy()
        self.resamplers[target_sample_freq] = resampler_f
        return resampler_f

    def __call__(self, x, target_sample_freq: float):
        if self.input_sample_freq == target_sample_freq:
            return x, target_sample_freq

        resampler = self.get_resampler(target_sample_freq)
        return resampler(x), target_sample_freq


class Any2AnyFreqResampler:
    def __init__(self):
        self.resamplers = {}

    def get_resampler(self, input_sample_freq, target_sample_freq):
        key = f"{input_sample_freq}-{target_sample_freq}"
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
        except:
            resampler = tat.Resample(
                int(input_sample_freq),
                int(target_sample_freq),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        resampler_f = lambda x: resampler(torch.from_numpy(x)).numpy()
        self.resamplers[input_sample_freq] = resampler_f
        return resampler_f

    def __call__(self, x, input_sample_freq: float, target_sample_freq: float):
        if input_sample_freq == target_sample_freq:
            return x, input_sample_freq

        resampler = self.get_resampler(input_sample_freq, target_sample_freq)
        dtype = x.dtype
        x = resampler(x.astype(np.float32, copy=False)).astype(dtype, copy=False)
        return x, target_sample_freq
