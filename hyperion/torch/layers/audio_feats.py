"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import math
import logging

from ...utils.misc import str2bool

import torch
import torch.nn as nn
import torch.cuda.amp as amp

try:
    from torch.fft import rfft as torch_rfft

    _rfft = lambda x: torch_rfft(x, dim=-1)
    _pow_spectrogram = lambda x: x.abs() ** 2
    _spectrogram = lambda x: x.abs()
except:
    _rfft = lambda x: torch.rfft(x, 1, normalized=False, onesided=True)
    _pow_spectrogram = lambda x: x.pow(2).sum(-1)
    _spectrogram = lambda x: x.pow(2).sum(-1).sqrt()

from ...feats.filter_banks import FilterBankFactory as FBF

# window types
HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]

# def _amp_safe_matmul(a, b):
#     if _use_amp():
#         mx = torch.max(a, dim=-1, keepdim=True)[0]
#         return mx*torch.matmul(a/mx, b)

#     return torch.matmul(a, b)


def _get_feature_window_function(window_type, window_size, blackman_coeff=0.42):
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=True)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=True, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        # return torch.hann_window(window_size, periodic=True).pow(0.85)
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return (0.5 - 0.5 * torch.cos(a * window_function)).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        )
    else:
        raise Exception("Invalid window type " + window_type)


def _get_strided_batch(waveform, window_length, window_shift, snip_edges, center=False):
    r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
    representing how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (torch.Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
        center (bool): If true, if puts the center of the frame at t*window_shift, starting at t=0,
                       If overwrides snip_edges and set it to False

    Returns:
        torch.Tensor: 3D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    num_samples = waveform.size(-1)
    if center:
        snip_edges = False

    if snip_edges:
        if num_samples < window_length:
            return torch.empty((0, 0, 0))
        else:
            num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        if center:
            npad_left = int(window_length // 2)
            npad_right = npad_left
            npad = 2 * npad_left
            num_frames = 1 + (num_samples + npad - window_length) // window_shift
        else:
            num_frames = (num_samples + (window_shift // 2)) // window_shift
            new_num_samples = (num_frames - 1) * window_shift + window_length
            npad = new_num_samples - num_samples
            npad_left = int((window_length - window_shift) // 2)
            npad_right = npad - npad_left

        # waveform = nn.functional.pad(waveform, (npad_left, npad_right), mode='reflect')
        pad_left = torch.flip(waveform[:, 1 : npad_left + 1], (1,))
        pad_right = torch.flip(waveform[:, -npad_right - 1 : -1], (1,))
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)

    strides = (
        waveform.stride(0),
        window_shift * waveform.stride(1),
        waveform.stride(1),
    )
    sizes = (batch_size, num_frames, window_length)
    return waveform.as_strided(sizes, strides)


def _get_log_energy(x, energy_floor):
    r"""Returns the log energy of size (m) for a strided_input (m,*)"""
    log_energy = (x.pow(2).sum(-1) + 1e-15).log()  # size (m)
    if energy_floor > 0.0:
        log_energy = torch.max(
            log_energy,
            torch.tensor(math.log(energy_floor), dtype=torch.get_default_dtype()),
        )

    return log_energy


class Wav2Win(nn.Module):
    def __init__(
        self,
        fs=16000,
        frame_length=25,
        frame_shift=10,
        pad_length=None,
        remove_dc_offset=True,
        preemph_coeff=0.97,
        window_type="povey",
        dither=1,
        snip_edges=True,
        center=False,
        energy_floor=0,
        raw_energy=True,
        return_log_energy=False,
    ):

        super().__init__()
        self.fs = fs
        self.frame_length = frame_length
        self.frame_shift = frame_shift

        self.remove_dc_offset = remove_dc_offset
        self.preemph_coeff = preemph_coeff
        self.window_type = window_type
        self.dither = dither
        self.snip_edges = snip_edges
        self.center = center
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.return_log_energy = return_log_energy

        N = int(math.floor(frame_length * fs / 1000))
        self._length = N
        self._shift = int(math.floor(frame_shift * fs / 1000))

        self._window = nn.Parameter(
            _get_feature_window_function(window_type, N), requires_grad=False
        )
        self.pad_length = N if pad_length is None else pad_length
        assert self.pad_length >= N

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = (
            "{}(fs={}, frame_length={}, frame_shift={}, pad_length={}, "
            "remove_dc_offset={}, preemph_coeff={}, window_type={} "
            "dither={}, snip_edges={}, center={}, energy_floor={}, raw_energy={}, return_log_energy={})"
        ).format(
            self.__class__.__name__,
            self.fs,
            self.frame_length,
            self.frame_shift,
            self.pad_length,
            self.remove_dc_offset,
            self.preemph_coeff,
            self.window_type,
            self.dither,
            self.snip_edges,
            self.center,
            self.energy_floor,
            self.raw_energy,
            self.return_log_energy,
        )
        return s

    def forward(self, x):

        # Add dither
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n

        # remove offset
        if self.remove_dc_offset:
            mu = torch.mean(x, dim=1, keepdim=True)
            x = x - mu

        if self.return_log_energy and self.raw_energy:
            # Compute the log energy of each frame
            x_strided = _get_strided_batch(
                x, self._length, self._shift, self.snip_edges, center=self.center
            )
            log_energy = _get_log_energy(x_strided, self.energy_floor)  # size (m)

        if self.preemph_coeff != 0.0:
            x_offset = torch.nn.functional.pad(
                x.unsqueeze(1), (1, 0), mode="replicate"
            ).squeeze(1)
            x = x - self.preemph_coeff * x_offset[:, :-1]

        x_strided = _get_strided_batch(
            x, self._length, self._shift, self.snip_edges, center=self.center
        )

        # Apply window_function to each frame
        x_strided = x_strided * self._window

        if self.return_log_energy and not self.raw_energy:
            signal_log_energy = _get_log_energy(
                strided_input, self.energy_floor
            )  # size (batch, m)

        # Pad columns with zero until we reach size (batch, num_frames, pad_length)
        if self.pad_length != self._length:
            pad = self.pad_length - self._length
            x_strided = torch.nn.functional.pad(
                x_strided.unsqueeze(1), (0, pad), mode="constant", value=0
            ).squeeze(1)

        if self.return_log_energy:
            return x_strided, log_energy

        return x_strided


class Wav2FFT(nn.Module):
    def __init__(
        self,
        fs=16000,
        frame_length=25,
        frame_shift=10,
        fft_length=512,
        remove_dc_offset=True,
        preemph_coeff=0.97,
        window_type="povey",
        dither=1,
        snip_edges=True,
        center=False,
        energy_floor=0,
        raw_energy=True,
        use_energy=True,
    ):

        super().__init__()

        N = int(math.floor(frame_length * fs / 1000))
        if N > fft_length:
            k = math.ceil(math.log(N) / math.log(2))
            self.fft_length = int(2 ** k)

        self.wav2win = Wav2Win(
            fs,
            frame_length,
            frame_shift,
            pad_length=fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            center=center,
            energy_floor=0,
            raw_energy=raw_energy,
            return_log_energy=use_energy,
        )

        self.fft_length = fft_length
        self.use_energy = use_energy

    @property
    def fs(self):
        return self.wav2win.fs

    @property
    def frame_length(self):
        return self.wav2win.frame_length

    @property
    def frame_shift(self):
        return self.wav2win.frame_shift

    @property
    def remove_dc_offset(self):
        return self.wav2win.remove_dc_offset

    @property
    def preemph_coeff(self):
        return self.wav2win.preemph_coeff

    @property
    def window_type(self):
        return self.wav2win.window_type

    @property
    def dither(self):
        return self.wav2win.dither

    def forward(self, x):

        x_strided = self.wav2win(x)
        if self.use_energy:
            x_strided, log_e = x_strided

        # X = torch.rfft(x_strided, 1, normalized=False, onesided=True)
        X = _rfft(x_strided)

        if self.use_energy:
            X[:, 0, :, 0] = log_e

        return X


class Wav2Spec(Wav2FFT):
    def __init__(
        self,
        fs=16000,
        frame_length=25,
        frame_shift=10,
        fft_length=512,
        remove_dc_offset=True,
        preemph_coeff=0.97,
        window_type="povey",
        use_fft_mag=False,
        dither=1,
        snip_edges=True,
        center=False,
        energy_floor=0,
        raw_energy=True,
        use_energy=True,
    ):

        super().__init__(
            fs,
            frame_length,
            frame_shift,
            fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            center=center,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def forward(self, x):

        x_strided = self.wav2win(x)
        if self.use_energy:
            x_strided, log_e = x_strided

        # X = torch.rfft(x_strided, 1, normalized=False, onesided=True)
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        # pow_spec = X.pow(2).sum(-1)
        # if self.use_fft_mag:
        #     pow_spec = pow_spec.sqrt()

        if self.use_energy:
            pow_spec[:, 0] = log_e

        return pow_spec


class Wav2LogSpec(Wav2FFT):
    def __init__(
        self,
        fs=16000,
        frame_length=25,
        frame_shift=10,
        fft_length=512,
        remove_dc_offset=True,
        preemph_coeff=0.97,
        window_type="povey",
        use_fft_mag=False,
        dither=1,
        snip_edges=True,
        center=False,
        energy_floor=0,
        raw_energy=True,
        use_energy=True,
    ):

        super().__init__(
            fs,
            frame_length,
            frame_shift,
            fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            center=center,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def forward(self, x):

        x_strided = self.wav2win(x)
        if self.use_energy:
            x_strided, log_e = x_strided

        # X = torch.rfft(x_strided, 1, normalized=False, onesided=True)
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)

        # pow_spec = X.pow(2).sum(-1)
        # if self.use_fft_mag:
        #     pow_spec = pow_spec.sqrt()

        pow_spec = (pow_spec + 1e-15).log()

        if self.use_energy:
            pow_spec[:, 0] = log_e

        return pow_spec


class Wav2LogFilterBank(Wav2FFT):
    def __init__(
        self,
        fs=16000,
        frame_length=25,
        frame_shift=10,
        fft_length=512,
        remove_dc_offset=True,
        preemph_coeff=0.97,
        window_type="povey",
        use_fft_mag=False,
        dither=1,
        fb_type="mel_kaldi",
        low_freq=20,
        high_freq=0,
        num_filters=23,
        norm_filters=False,
        snip_edges=True,
        center=False,
        energy_floor=0,
        raw_energy=True,
        use_energy=True,
    ):

        super().__init__(
            fs,
            frame_length,
            frame_shift,
            fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            center=center,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        self.fb_type = fb_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters

        fb = FBF.create(
            fb_type,
            num_filters,
            self.fft_length,
            self.fs,
            low_freq,
            high_freq,
            norm_filters,
        )
        self._fb = nn.Parameter(
            torch.tensor(fb, dtype=torch.get_default_dtype()), requires_grad=False
        )
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def forward(self, x):

        x_strided = self.wav2win(x)
        if self.use_energy:
            x_strided, log_e = x_strided

        # X = torch.rfft(x_strided, 1, normalized=False, onesided=True)
        X = _rfft(x_strided)
        # logging.info('X={} {}'.format(X, X.type()))
        # logging.info('X={}'.format(X.type()))
        pow_spec = self._to_spec(X)
        # pow_spec = X.pow(2).sum(-1)
        # # logging.info('p={} {} nan={}'.format(pow_spec, pow_spec.type(), torch.sum(torch.isnan(pow_spec))))
        # # logging.info('p={}'.format(pow_spec.type()))
        # if self.use_fft_mag:
        #     pow_spec = pow_spec.sqrt()

        with amp.autocast(enabled=False):
            pow_spec = torch.matmul(pow_spec.float(), self._fb.float())
        # logging.info('fb={} {}'.format(pow_spec, pow_spec.type()))
        # logging.info('fb={}'.format(pow_spec.type()))
        pow_spec = (pow_spec + 1e-10).log()
        # logging.info('lfb={} {}'.format(pow_spec, pow_spec.type()))
        # logging.info('lfb={}'.format(pow_spec.type()))
        if self.use_energy:
            pow_spec = torch.cat((log_e.unsqueeze(-1), pow_spec), dim=-1)

        return pow_spec


class Wav2MFCC(Wav2FFT):
    def __init__(
        self,
        fs=16000,
        frame_length=25,
        frame_shift=10,
        fft_length=512,
        remove_dc_offset=True,
        preemph_coeff=0.97,
        window_type="povey",
        use_fft_mag=False,
        dither=1,
        fb_type="mel_kaldi",
        low_freq=20,
        high_freq=0,
        num_filters=23,
        norm_filters=False,
        num_ceps=13,
        snip_edges=True,
        center=False,
        cepstral_lifter=22,
        energy_floor=0,
        raw_energy=True,
        use_energy=True,
    ):

        super().__init__(
            fs,
            frame_length,
            frame_shift,
            fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            center=center,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        self.fb_type = fb_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self.num_ceps = num_ceps
        self.cepstral_lifter = cepstral_lifter

        fb = FBF.create(
            fb_type,
            num_filters,
            self.fft_length,
            self.fs,
            low_freq,
            high_freq,
            norm_filters,
        )
        self._fb = nn.Parameter(
            torch.tensor(fb, dtype=torch.get_default_dtype()), requires_grad=False
        )
        self._dct = nn.Parameter(
            self.make_dct_matrix(self.num_ceps, self.num_filters), requires_grad=False
        )
        self._lifter = nn.Parameter(
            self.make_lifter(self.num_ceps, self.cepstral_lifter), requires_grad=False
        )
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

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
        return 1 + 0.5 * Q * torch.sin(
            math.pi * torch.arange(N, dtype=torch.get_default_dtype()) / Q
        )

    @staticmethod
    def make_dct_matrix(num_ceps, num_filters):
        n = torch.arange(float(num_filters)).unsqueeze(1)
        k = torch.arange(float(num_ceps))
        dct = torch.cos(
            math.pi / float(num_filters) * (n + 0.5) * k
        )  # size (n_mfcc, n_mels)
        dct[:, 0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(num_filters))
        return dct

    def forward(self, x):

        x_strided = self.wav2win(x)
        if self.use_energy:
            x_strided, log_e = x_strided

        # X = torch.rfft(x_strided, 1, normalized=False, onesided=True)
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        # pow_spec = X.pow(2).sum(-1)
        # if self.use_fft_mag:
        #     pow_spec = pow_spec.sqrt()

        with amp.autocast(enabled=False):
            pow_spec = torch.matmul(pow_spec.float(), self._fb.float())

        pow_spec = (pow_spec + 1e-10).log()

        mfcc = torch.matmul(pow_spec, self._dct)
        if self.cepstral_lifter > 0:
            mfcc *= self._lifter

        if self.use_energy:
            mfcc[:, 0] = log_e

        return mfcc


class Wav2KanBayashiLogFilterBank(Wav2LogFilterBank):
    """Class to replicate log-filter-banks used in
    Kan Bayashi's ParallelWaveGAN repository:
    https://github.com/kan-bayashi/ParallelWaveGAN
    """

    def __init__(
        self,
        fs=16000,
        frame_length=64,
        frame_shift=16,
        fft_length=1024,
        remove_dc_offset=True,
        window_type="hanning",
        low_freq=80,
        high_freq=7600,
        num_filters=80,
        snip_edges=False,
        center=True,
    ):

        super().__init__(
            fs=fs,
            frame_length=frame_length,
            frame_shift=frame_shift,
            fft_length=fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=0,
            window_type=window_type,
            use_fft_mag=True,
            dither=1e-5,
            fb_type="mel_librosa",
            low_freq=low_freq,
            high_freq=high_freq,
            num_filters=num_filters,
            norm_filters=True,
            snip_edges=snip_edges,
            center=center,
            use_energy=False,
        )

        # Kan Bayashi uses log10 instead of log
        self.scale = 1.0 / math.log(10)

    def forward(self, x):
        return self.scale * super().forward(x)


class Spec2LogFilterBank:
    def __init__(
        self,
        fs=16000,
        fft_length=512,
        fb_type="mel_kaldi",
        low_freq=20,
        high_freq=0,
        num_filters=23,
        norm_filters=False,
    ):

        super().__init__()
        self.fs = fs
        self.fft_length = fft_length
        self.fb_type = fb_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters

        fb = FBF.create(
            fb_type,
            num_filters,
            self.fft_length,
            self.fs,
            low_freq,
            high_freq,
            norm_filters,
        )
        self._fb = nn.Parameter(
            torch.tensor(fb, dtype=torch.get_default_dtype()), requires_grad=False
        )

    def forward(self, x):
        with amp.autocast(enabled=False):
            pow_spec = torch.matmul(x.float(), self._fb.float())
        pow_spec = (pow_spec + 1e-10).log()
        return pow_spec
