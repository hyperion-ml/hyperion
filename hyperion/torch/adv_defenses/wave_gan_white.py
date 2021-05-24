# Added wave_gan_model_ckpt to test using different model ckpts [Sonal 24Aug20]

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import parallel_wavegan.models
import torch
import yaml
from parallel_wavegan.layers import PQMF
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.utils import read_hdf5
from sklearn.preprocessing import StandardScaler
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class WaveGANReconstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, audio, feature_extractor, wave_gan, pqmf, use_noise_input, config, device, pad_fn):
        """
        WaveGAN Vocoder signal reconstruction from spectrum. In principle this can be backpropagated through,
        but our current implementation disallows that.
        """
        # TODO: remove librosa and use pytorch feature extraction for batch processing and backprop
        reconstructions = []
        num_samples = audio.shape[1]
        for idx in range(audio.shape[0]):
            recording = audio[idx, :].detach().cpu().numpy()
            mel_spectrogram = feature_extractor.transform(recording)
            # Setup inputs
            with torch.no_grad():
                inputs = ()
                if use_noise_input:
                    noise = torch.randn(1, 1, len(mel_spectrogram) * config["hop_size"]).to(device)
                    inputs += (noise,)
                mel_spectrogram = pad_fn(
                    torch.from_numpy(mel_spectrogram).unsqueeze(0).transpose(2, 1)
                ).to(device)
                inputs += (mel_spectrogram,)
                # Generate
                if config["generator_params"]["out_channels"] == 1:
                    reconstructed_audio = wave_gan(*inputs).view(-1)
                    reconstructed_audio = reconstructed_audio[:num_samples]
                else:
                    reconstructed_audio = pqmf.synthesis(wave_gan(*inputs)).view(-1)
                    reconstructed_audio = reconstructed_audio[:, :num_samples]
                reconstructions.append(reconstructed_audio)
        return torch.stack(reconstructions)

    @staticmethod
    def backward(ctx, grad_output):
        """Since WaveGAN is not diffentiable with the current implementation, we define grad_x=1
        """
        # we need one output per input
        return grad_output.clone(), None, None, None, None, None, None, None


class WaveGANDefender(nn.Module):
    def __init__(self, wave_gan_model_dir : Path, wave_gan_model_ckpt : Path,  device: str = DEVICE):
        super().__init__()
        self.device = device
        with open(wave_gan_model_dir / 'config.yml') as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self.feature_extractor = WaveGANFeatureExtractor(wave_gan_model_dir)

        self.model = ParallelWaveGANGenerator(**self.config["generator_params"])
        self.model.load_state_dict(
            torch.load(wave_gan_model_dir / wave_gan_model_ckpt , map_location=device)["model"]["generator"]
        )
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(device)

        self.use_noise_input = not isinstance(self.model, parallel_wavegan.models.MelGANGenerator)
        self.pad_fn = torch.nn.ReplicationPad1d(self.config["generator_params"].get("aux_context_window", 0))
        if self.config["generator_params"]["out_channels"] > 1:
            self.pqmf = PQMF(self.config["generator_params"]["out_channels"]).to(device)
        else:
            self.pqmf = None

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return WaveGANReconstruction.apply(
            audio,
            self.feature_extractor,
            self.model,
            self.pqmf,
            self.use_noise_input,
            self.config,
            self.device,
            self.pad_fn
        )


def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


class WaveGANFeatureExtractor:
    def __init__(self, wave_gan_model_dir: Path):
        with open(wave_gan_model_dir / 'config.yml') as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        # Restore scaler
        scaler = StandardScaler()
        stats_path = str(wave_gan_model_dir / 'stats.h5')
        if self.config["format"] == "hdf5":
            scaler.mean_ = read_hdf5(stats_path, "mean")
            scaler.scale_ = read_hdf5(stats_path, "scale")
        elif self.config["format"] == "npy":
            scaler.mean_ = np.load(stats_path)[0]
            scaler.scale_ = np.load(stats_path)[1]
        else:
            raise ValueError("support only hdf5 or npy format.")
        # from version 0.23.0, this information is needed
        scaler.n_features_in_ = scaler.mean_.shape[0]
        self.scaler = scaler

    def transform(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=self.config["trim_threshold_in_db"],
                frame_length=self.config["trim_frame_size"],
                hop_length=self.config["trim_hop_size"]
            )

        mel = logmelfilterbank(
            audio,
            sampling_rate=self.config["sampling_rate"],
            hop_size=self.config["hop_size"],
            fft_size=self.config["fft_size"],
            win_length=self.config["win_length"],
            window=self.config["window"],
            num_mels=self.config["num_mels"],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"]
        )

        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, self.config["fft_size"]), mode="edge")
        audio = audio[:len(mel) * self.config["hop_size"]]
        assert len(mel) * self.config["hop_size"] == len(audio)

        # Normalize the mel spectrogram
        mel = self.scaler.transform(mel)

        return mel
