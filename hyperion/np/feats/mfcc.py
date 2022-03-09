"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from enum import Enum

import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter

from ...hyp_defs import float_cpu
from ...utils.misc import str2bool
from .feature_windows import FeatureWindowFactory as FWF
from .filter_banks import FilterBankFactory as FBF
from .stft import strft, st_logE


class MFCCSteps(Enum):
    """Steps in the MFCC pipeline"""

    WAVE = 0
    FFT = 1
    SPEC = 2
    LOG_SPEC = 3
    LOGFB = 4
    MFCC = 5

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other):
        if self.__class__ is other.__class__:
            return self.value != other.value
        return NotImplemented


mfcc_steps_dict = {step.name.lower(): step for step in MFCCSteps}


class MFCC(object):
    """Compute MFCC features.

    Attributes:
       sample_frequency:  Waveform data sample frequency (must match the waveform file, if specified there) (default = 16000)
       frame_length:      Frame length in milliseconds (default = 25)
       frame_shift:       Frame shift in milliseconds (default = 10)
       fft_length:        Length of FFT (default = 512)
       remove_dc_offset:  Subtract mean from waveform on each frame (default = True)
       preemphasis_coeff: Coefficient for use in signal preemphasis (default = 0.97)
       window_type:       Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"blackmann") (default = 'povey')
       use_fft2:          If true, it uses |X(f)|^2, if false, it uses |X(f)|, (default = True)
       dither:            Dithering constant (0.0 means no dither) (default = 1)
       fb_type:           Filter-bank type: mel_kaldi, mel_etsi, mel_librosa, mel_librosa_htk, linear (default = 'mel_kaldi')
       low_freq:          Low cutoff frequency for mel bins (default = 20)
       high_freq:         High cutoff frequency for mel bins (if < 0, offset from Nyquist) (default = 0)
       num_filters:       Number of triangular mel-frequency bins (default = 23)
       norm_filters:      Normalize filters coeff to sum up to 1, if librosa it uses stanley norm (default = False)
       num_ceps:          Number of cepstra in MFCC computation (including C0) (default = 13)
       snip_edges:        If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (default = True)
       energy_floor:      Floor on energy (absolute, not relative) in MFCC computation (default = 0)
       raw_energy:        If true, compute energy before preemphasis and windowing (default = True)
       use_energy:        Use energy (not C0) in MFCC computation (default = True)
       cepstral_lifter:   Constant that controls scaling of MFCCs (default = 22)
       input_step:        It can continue computation from any step: wav, fft, spec, logfb (default = 'wav')
       output_step:       It can return intermediate result: fft, spec, logfb, mfcc (default = 'mfcc')
    """

    def __init__(
        self,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
        fft_length=512,
        remove_dc_offset=True,
        preemphasis_coeff=0.97,
        window_type="povey",
        use_fft2=True,
        dither=1,
        fb_type="mel_kaldi",
        low_freq=20,
        high_freq=0,
        num_filters=23,
        norm_filters=False,
        num_ceps=13,
        snip_edges=True,
        energy_floor=0,
        raw_energy=True,
        use_energy=True,
        cepstral_lifter=22,
        input_step="wave",
        output_step="mfcc",
    ):

        self.fs = sample_frequency
        self.sample_frequency = sample_frequency
        fs = self.fs
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fft_length = fft_length
        self.remove_dc_offset = remove_dc_offset
        self.preemphasis_coeff = preemphasis_coeff
        self.window_type = window_type
        # self.blackman_coeff = blackman_coeff
        self.use_fft2 = use_fft2
        self.dither = dither
        self.fb_type = fb_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self.num_ceps = num_ceps
        self.snip_edges = snip_edges
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.use_energy = use_energy
        self.cepstral_lifter = cepstral_lifter
        self.input_step = input_step
        self.output_step = output_step

        assert input_step in mfcc_steps_dict, "Invalid input step %s" % (input_step)
        assert output_step in mfcc_steps_dict, "Invalid output step %s" % (output_step)

        self._input_step = mfcc_steps_dict[input_step]
        self._output_step = mfcc_steps_dict[output_step]

        N = int(np.floor(frame_length * fs / 1000))
        if N > fft_length:
            k = np.ceil(np.log(N) / np.log(2))
            self.fft_length = int(2 ** k)

        self._length = N
        self._shift = int(np.floor(frame_shift * fs / 1000))

        self._dc_a = np.array([1, -0.999], dtype=float_cpu())
        self._dc_b = np.array([1, -1], dtype=float_cpu())

        self._preemph_b = np.array([1, -self.preemphasis_coeff], dtype=float_cpu())

        self._window = FWF.create(window_type, N)
        # corrects scipy.stft scales fft by 1/sum(window)
        # self._fft_scale = np.sum(self._window)
        self._fb = FBF.create(
            fb_type, num_filters, self.fft_length, fs, low_freq, high_freq, norm_filters
        )
        self._lifter = MFCC.make_lifter(self.num_ceps, self.cepstral_lifter)
        self.reset()

    def reset(self):
        """Resets the internal states of the filters"""
        self._dc_zi = np.array([0], dtype=float_cpu())
        self._preemph_zi = np.array([0], dtype=float_cpu())

    @staticmethod
    def make_lifter(N, Q):
        """Makes the liftering function

        Args:
          N: Number of cepstral coefficients.
          Q: Liftering parameter

        Returns:
          Liftering vector.
        """
        if Q == 0:
            return 1
        return 1 + 0.5 * Q * np.sin(np.pi * np.arange(N) / Q)

    def compute_raw_logE(self, x):
        """Computes log-energy before preemphasis filter

        Args:
          x: wave signal

        Returns:
          Log-energy
        """
        return st_logE(x, self._length, self._shift)

    def compute(self, x, return_fft=False, return_spec=False, return_logfb=False):
        """Evaluates the MFCC pipeline.

        Args:
          x:               Wave, stft, spectrogram or log-filter-bank depending on input_step.
          return_fft:      If true, it also returns short-time fft.
          return_spec:  If true, it also returns short-time magnitude spectrogram.
          return_logfb:    If true, it also returns log-filter-bank.

        Returns:
          Stfft, spectrogram, log-filter-bank or MFCC depending on output_step.
        """

        assert not (return_fft and self._input_step > MFCCSteps.FFT)
        assert not (
            return_spec
            and (
                self._input_step > MFCCSteps.SPEC or self._output_step < MFCCSteps.SPEC
            )
        )
        assert not (return_logfb and self._output_step < MFCCSteps.LOGFB)

        # Prepare input
        if self._input_step == MFCCSteps.FFT:
            X = x
            F = np.abs(X)
            if self.use_energy:
                logE = F[:, 0]
        elif self._input_step == MFCCSteps.SPEC:
            F = x
            if self.use_energy:
                logE = F[:, 0]
        elif self._input_step == MFCCSteps.LOG_SPEC:
            if self.use_energy:
                logE = x[:, 0]
            F = np.exp(x)
        elif self._input_step == MFCCSteps.LOGFB:
            if self.use_energy:
                B = x[:, 1:]
                logE = x[:, 0]

        if self._input_step == MFCCSteps.WAVE:
            if self.snip_edges:
                num_frames = int(
                    np.floor((len(x) - self._length + self._shift) / self._shift)
                )
            else:
                num_frames = int(np.round(len(x) / self._shift))
                len_x = (num_frames - 1) * self._shift + self._length
                dlen_x = len_x - len(x)
                # x = np.pad(x, (0, dlen_x), mode='reflect')
                dlen1_x = int(np.floor((self._length - self._shift) / 2))
                dlen2_x = int(dlen_x - dlen1_x)
                x = np.pad(x, (dlen1_x, dlen2_x), mode="reflect")

            # add dither
            if self.dither > 0:
                n = self.dither * np.random.RandomState(seed=len(x)).randn(
                    len(x)
                ).astype(float_cpu(), copy=False)
                x = x + n

            # Remove offset
            if self.remove_dc_offset:
                x, self._dc_zi = lfilter(self._dc_b, self._dc_a, x, zi=self._dc_zi)

            # Compute raw energy
            if self.use_energy and self.raw_energy:
                logE = self.compute_raw_logE(x)

            # Apply preemphasis filter
            if self.preemphasis_coeff > 0:
                x, self._preemph_zi = lfilter(
                    self._preemph_b, [1], x, zi=self._preemph_zi
                )

            # Comptue STFFT
            # _, _, X = stft(x, window=self._window, nperseg=self._nperseg, noverlap=self._overlap, nfft=self.fft_length, boundary=None)
            # Fix scale of FFT
            # X = self._fft_scale * X[:, :num_frames].T
            # xx = []
            # j = 0
            # for i in range(len(x)//160-2):
            #     #print(x[j:j+400].shape)
            #     #print(self._window.shape)
            #     xx.append(x[j:j+400]*self._window)
            #     j += 160

            # return np.vstack(tuple(xx))

            X = strft(x, self._length, self._shift, self.fft_length, self._window)

            # Compute |X(f)|
            F = np.abs(X).astype(dtype=float_cpu(), copy=False)

            # Compute no-raw energy
            if self.use_energy and not self.raw_energy:
                # Use Paserval's theorem
                logE = np.log(np.mean(F ** 2, axis=-1) + 1e-10)

        # Compute |X(f)|^2
        if self._input_step <= MFCCSteps.FFT and self._output_step >= MFCCSteps.SPEC:
            if self.use_fft2:
                F = F ** 2

        # Compute log-filter-bank
        if (
            self._input_step <= MFCCSteps.LOG_SPEC
            and self._output_step >= MFCCSteps.LOGFB
        ):
            B = np.log(np.dot(F, self._fb) + 1e-10)
            # B = np.maximum(B, np.log(self.energy_floor+1e-15))

        # Compute MFCC
        if self._input_step <= MFCCSteps.LOGFB and self._output_step == MFCCSteps.MFCC:
            P = dct(B, type=2, norm="ortho")[:, : self.num_ceps]

            if self.cepstral_lifter > 0:
                P *= self._lifter

        # Select the right output type
        if self._output_step == MFCCSteps.FFT:
            R = X
        elif self._output_step == MFCCSteps.SPEC:
            R = F
        elif self._output_step == MFCCSteps.LOG_SPEC:
            R = np.log(F + 1e-10)
        elif self._output_step == MFCCSteps.LOGFB:
            R = B
        else:
            R = P

        if self.use_energy:
            # append energy
            logE = np.maximum(logE, np.log(self.energy_floor + 1e-15))
            if self._output_step == MFCCSteps.LOGFB:
                R = np.hstack((logE[:, None], R))
            else:
                R[:, 0] = logE

        if not (return_fft or return_spec or return_logfb):
            return R

        # Append fft, fft magnitude, log-filter-bank
        R = [R]
        if return_fft:
            R = R + [X]
        if return_spec:
            R = R + [F]
        if return_logfb:
            R = R + [B]

        return tuple(R)

    @staticmethod
    def filter_args(**kwargs):
        """Filters MFCC args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with MFCC options.
        """
        valid_args = (
            "sample_frequency",
            "frame_length",
            "frame_shift",
            "fft_length",
            "remove_dc_offset",
            "preemphasis_coeff",
            "window_type",
            "blackman_coeff",
            "use_fft2",
            "dither",
            "fb_type",
            "low_freq",
            "high_freq",
            "num_filters",
            "norm_filters",
            "num_ceps",
            "snip_edges",
            "energy_floor",
            "raw_energy",
            "use_energy",
            "cepstral_lifter",
            "input_step",
            "output_step",
        )

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds MFCC options to parser.

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
            help="Waveform data sample frequency "
            "(must match the waveform file, if specified there)",
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
            p1 + "fft-length", type=int, default=512, help="Length of FFT"
        )

        parser.add_argument(
            p1 + "remove-dc-offset",
            default=True,
            type=str2bool,
            help="Subtract mean from waveform on each frame",
        )

        parser.add_argument(
            p1 + "preemphasis-coeff",
            type=float,
            default=0.97,
            help="Coefficient for use in signal preemphasis",
        )

        FWF.add_class_args(parser, prefix)

        parser.add_argument(
            p1 + "use-fft2",
            default=True,
            type=str2bool,
            help="If true, it uses |X(f)|^2, if false, it uses |X(f)|",
        )

        parser.add_argument(
            p1 + "dither",
            type=float,
            default=1,
            help="Dithering constant (0.0 means no dither)",
        )

        FBF.add_class_args(parser, prefix)

        parser.add_argument(
            p1 + "num-ceps",
            type=int,
            default=13,
            help="Number of cepstra in MFCC computation (including C0)",
        )

        parser.add_argument(
            p1 + "snip-edges",
            default=True,
            type=str2bool,
            help=(
                "If true, end effects will be handled by outputting "
                "only frames that completely fit in the file, and the "
                "number of frames depends on the frame-length. "
                "If false, the number of frames depends only on the "
                "frame-shift, and we reflect the data at the ends."
            ),
        )

        parser.add_argument(
            p1 + "energy-floor",
            type=float,
            default=0,
            help="Floor on energy (absolute, not relative) in MFCC computation",
        )

        parser.add_argument(
            p1 + "raw-energy",
            default=True,
            type=str2bool,
            help="If true, compute energy before preemphasis and windowing",
        )
        parser.add_argument(
            p1 + "use-energy",
            default=True,
            type=str2bool,
            help="Use energy (not C0) in MFCC computation",
        )

        parser.add_argument(
            p1 + "cepstral-lifter",
            type=float,
            default=22,
            help="Constant that controls scaling of MFCCs",
        )

        parser.add_argument(
            p1 + "input-step",
            default="wave",
            choices=["wave", "fft", "spec", "log_spec", "logfb"],
            help=(
                "It can continue computation from any step: " "wav, fft, spec, logfb"
            ),
        )

        parser.add_argument(
            p1 + "output-step",
            default="mfcc",
            choices=["fft", "spec", "log_spec", "logfb", "mfcc"],
            help=(
                "It can return intermediate result: " "fft, spec, log_spec, logfb, mfcc"
            ),
        )

    add_argparse_args = add_class_args
