"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


class Resampler:
    def __init__(self, target_sample_freq: float):
        self.target_sample_freq = target_sample_freq
        self.resamplers = {}

    def _get_resampler(self, input_sample_freq):
        if input_sample_freq in self.resamplers:
            return self.resamplers[input_sample_freq]

        import torch
        import torchaudio.transforms as tat

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
        self.resamplers[fs] = resampler_f
        return resampler_f

    def __call__(self, x, sample_freq: float):
        if sample_freq == self.target_sample_freq:
            return x, sample_freq

        resampler = self._get_resampler(sample_freq)
        return resampler(x), self.target_sample_freq
