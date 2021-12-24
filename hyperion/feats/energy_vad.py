"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging

import numpy as np
from scipy.signal import lfilter

from ..hyp_defs import float_cpu
from ..utils.misc import str2bool
from .stft import st_logE


class EnergyVAD(object):
    """Compute VAD based on Kaldi Energy VAD method.

    Attributes:
       sample_frequency:                    Waveform data sample frequency (must match the waveform file, if specified there) (default = 16000)
       frame_length:          Frame length in milliseconds (default = 25)
       frame_shift:           Frame shift in milliseconds (default = 10)
       dither:                Dithering constant (0.0 means no dither) (default = 1)
       snip_edges:            If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (default = True)
       vad_energy_mean_scale: If this is set to s, to get the actual threshold we let m be the mean log-energy of the file, and use s*m + vad-energy-threshold (float, default = 0.5)
       vad_energy_threshold:  Constant term in energy threshold for MFCC0 for VAD (also see --vad-energy-mean-scale) (float, default = 5)
       vad_frames_context:    Number of frames of context on each side of central frame, in window for which energy is monitored (int, default = 0)
       vad_proportion_threshold: Parameter controlling the proportion of frames within the window that need to have more energy than the threshold (float, default = 0.6)
    """

    def __init__(
        self,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
        dither=1,
        snip_edges=True,
        vad_energy_mean_scale=0.5,
        vad_energy_threshold=5,
        vad_frames_context=0,
        vad_proportion_threshold=0.6,
    ):

        self.sample_frequency = sample_frequency
        fs = sample_frequency
        self.fs = fs
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.snip_edges = snip_edges

        N = int(np.floor(frame_length * fs / 1000))
        self._length = N
        self._shift = int(np.floor(frame_shift * fs / 1000))

        self._dc_a = np.array([1, -0.999], dtype=float_cpu())
        self._dc_b = np.array([1, -1], dtype=float_cpu())

        assert vad_energy_mean_scale >= 0
        assert vad_frames_context >= 0
        assert vad_proportion_threshold > 0 and vad_proportion_threshold < 1

        self.vad_energy_mean_scale = vad_energy_mean_scale
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_frames_context = vad_frames_context
        self.vad_proportion_threshold = vad_proportion_threshold

        self.reset()

    def reset(self):
        """Resets the internal states of the filters"""
        self._dc_zi = np.array([0], dtype=float_cpu())

    def compute(self, x, return_loge=False):
        """Evaluates the VAD.

        Args:
          x:               Wave
          return_loge:     If true, it also returns the log-energy.

        Returns:
          Binary VAD
        """

        if x.ndim == 1:
            # Input is wave
            if self.snip_edges:
                num_frames = int(
                    np.floor((len(x) - self._length + self._shift) / self._shift)
                )
            else:
                num_frames = int(np.round(len(x) / self._shift))
                len_x = (num_frames - 1) * self._shift + self._length
                dlen_x = len_x - len(x)
                dlen1_x = int(np.floor((self._length - self._shift) / 2))
                dlen2_x = int(dlen_x - dlen1_x)
                x = np.pad(x, (dlen1_x, dlen2_x), mode="reflect")

            # add dither
            if self.dither > 0:
                n = self.dither * np.random.RandomState(seed=len(x)).randn(
                    len(x)
                ).astype(float_cpu(), copy=False)
                x = x + n

            x, self._dc_zi = lfilter(self._dc_b, self._dc_a, x, zi=self._dc_zi)

            # Compute raw energy
            logE = st_logE(x, self._length, self._shift)
        elif x.ndim == 2:
            # Assume that input are features with log-e in the first coeff of the vector
            logE = x[:, 0]
        else:
            raise Exception("Wrong input dimension ndim=%d" % x.ndim)

        # compute VAD from logE
        # print(np.mean(logE))
        e_thr = self.vad_energy_threshold + self.vad_energy_mean_scale * np.mean(logE)
        # print(e_thr)
        # print(logE)
        vad = logE > e_thr

        context = self.vad_frames_context
        if context == 0:
            return vad

        window = 2 * context + 1
        if len(vad) < window:
            context = int(len(vad) - 1 / 2)
            window = 2 * context + 1

        if window == 1:
            return vad

        h = np.ones((window,), dtype="float32")
        num_count = np.convolve(vad.astype("float32"), h, "same")
        den_count_boundary = np.arange(context + 1, window, dtype="float32")
        num_count[:context] /= den_count_boundary
        num_count[-context:] /= den_count_boundary[::-1]
        num_count[context:-context] /= window

        vad = num_count > self.vad_proportion_threshold
        return vad

    @staticmethod
    def filter_args(**kwargs):
        """Filters VAD args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with VAD options.
        """
        valid_args = (
            "sample_frequency",
            "frame_length",
            "frame_shift",
            "dither",
            "snip_edges",
            "vad_energy_mean_scale",
            "vad_energy_threshold",
            "vad_frames_context",
            "vad_proportion_threshold",
        )

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds VAD options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """

        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "sample-frequency",
            default=16000,
            type=int,
            help=(
                "Waveform data sample frequency "
                "(must match the waveform file, if specified there)"
            ),
        )

        parser.add_argument(
            p1 + "frame-length",
            type=int,
            default=25,
            help="Frame length in milliseconds",
        )
        parser.add_argument(
            p1 + "frame-shift", type=int, default=10, help="Frame shift in milliseconds"
        )

        parser.add_argument(
            p1 + "dither",
            type=float,
            default=1,
            help="Dithering constant (0.0 means no dither)",
        )

        parser.add_argument(
            p1 + "snip-edges",
            default=True,
            type=str2bool,
            help=(
                "If true, end effects will be handled by outputting only "
                "frames that completely fit in the file, and the number of "
                "frames depends on the frame-length. "
                "If false, the number of frames depends only on the "
                "frame-shift, and we reflect the data at the ends."
            ),
        )

        parser.add_argument(
            p1 + "vad-energy-mean-scale",
            type=float,
            default=0.5,
            help=(
                "If this is set to s, to get the actual threshold we let m "
                "be the mean log-energy of the file, and use "
                "s*m + vad-energy-threshold"
            ),
        )
        parser.add_argument(
            p1 + "vad-energy-threshold",
            type=float,
            default=5,
            help="Constant term in energy threshold for MFCC0 for VAD",
        )
        parser.add_argument(
            p1 + "vad-frames-context",
            type=int,
            default=0,
            help=(
                "Number of frames of context on each side of central frame, "
                "in window for which energy is monitored"
            ),
        )
        parser.add_argument(
            p1 + "vad-proportion-threshold",
            type=float,
            default=0.6,
            help=(
                "Parameter controlling the proportion of frames within "
                "the window that need to have more energy than the threshold"
            ),
        )

    add_argparse_args = add_class_args
