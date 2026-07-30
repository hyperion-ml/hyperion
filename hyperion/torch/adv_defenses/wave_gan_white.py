# Author : Piotr Zelasko
# Added wave_gan_model_ckpt to test using different model ckpts [Sonal 24Aug20]

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import librosa
import numpy as np
import torch
import yaml

try:
    # import parallel_wavegan.models
    from parallel_wavegan.layers import PQMF
    from parallel_wavegan.models import ParallelWaveGANGenerator
    from parallel_wavegan.utils import read_hdf5
except Exception:
    pass

from sklearn.preprocessing import StandardScaler
from torch import nn


class WaveGANReconstruction(nn.Module):
    """Reconstruct waveforms from mel features using a Parallel WaveGAN vocoder."""

    def __init__(
        self,
        feature_extractor: "WaveGANFeatureExtractor",
        wave_gan: nn.Module,
        pqmf: Optional[nn.Module],
        use_noise_input: bool,
        config: Mapping[str, Any],
        pad_fn: nn.Module,
    ) -> None:
        """Initialize waveform reconstructor.

        Args:
            feature_extractor: Mel feature extractor module.
            wave_gan: Parallel WaveGAN generator.
            pqmf: Optional PQMF synthesis module for multiband output.
            use_noise_input: Whether to prepend random noise input to the generator.
            config: WaveGAN configuration dictionary.
            pad_fn: Padding module for generator context.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.wave_gan = wave_gan
        self.pqmf = pqmf
        self.use_noise_input = use_noise_input
        self.config = config
        self.pad_fn = pad_fn

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Reconstruct one or more waveforms from input audio.

        Args:
            audio: Audio tensor shaped ``(T,)`` or ``(B, T)``.

        Returns:
            Reconstructed audio tensor with matching sample length.
        """
        feature_extractor = self.feature_extractor
        wave_gan = self.wave_gan
        pqmf = self.pqmf
        use_noise_input = self.use_noise_input
        config = self.config
        pad_fn = self.pad_fn

        # Added for processing single audio file as in deepspeech armory [Sonal 29Oct20]
        if audio.ndim == 1:
            num_samples = audio.shape[0]
            mel_spectrogram = feature_extractor.transform(audio)
            # Setup inputs
            inputs = ()
            if use_noise_input:
                noise = torch.randn(
                    1,
                    1,
                    len(mel_spectrogram) * config["hop_size"],
                    device=mel_spectrogram.device,
                )
                inputs += (noise,)

            mel_spectrogram = pad_fn(mel_spectrogram.unsqueeze(0).transpose(2, 1))
            inputs += (mel_spectrogram,)
            # Generate
            if config["generator_params"]["out_channels"] == 1:
                reconstructed_audio = wave_gan(*inputs).view(-1)
                reconstructed_audio = reconstructed_audio[:num_samples]
            else:
                reconstructed_audio = pqmf.synthesis(wave_gan(*inputs)).view(-1)
                reconstructed_audio = reconstructed_audio[:num_samples]
            return reconstructed_audio

        else:
            reconstructions = []
            num_samples = audio.shape[1]
            for idx in range(audio.shape[0]):
                recording = audio[idx, :]
                mel_spectrogram = feature_extractor.transform(recording)
                # Setup inputs
                inputs = ()
                if use_noise_input:
                    noise = torch.randn(
                        1,
                        1,
                        len(mel_spectrogram) * config["hop_size"],
                        device=recording.device,
                    )
                    inputs += (noise,)
                mel_spectrogram = pad_fn(mel_spectrogram.unsqueeze(0).transpose(2, 1))
                inputs += (mel_spectrogram,)
                # Generate
                if config["generator_params"]["out_channels"] == 1:
                    reconstructed_audio = wave_gan(*inputs).view(-1)
                    reconstructed_audio = reconstructed_audio[:num_samples]
                else:
                    reconstructed_audio = pqmf.synthesis(wave_gan(*inputs)).view(-1)
                    reconstructed_audio = reconstructed_audio[:num_samples]
                reconstructions.append(reconstructed_audio)
            return torch.stack(reconstructions)


class WaveGANDefender(nn.Module):
    """Defense module that denoises audio via WaveGAN reconstruction."""

    def __init__(self, wave_gan_model_dir: Path, wave_gan_model_ckpt: Path) -> None:
        """Load WaveGAN assets and build the reconstruction pipeline.

        Args:
            wave_gan_model_dir: Directory containing config and feature stats.
            wave_gan_model_ckpt: Generator checkpoint path relative to
                ``wave_gan_model_dir`` (or absolute path).
        """
        super().__init__()
        with open(wave_gan_model_dir / "config.yml") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self.feature_extractor = WaveGANFeatureExtractor(wave_gan_model_dir)

        self.model = ParallelWaveGANGenerator(**self.config["generator_params"])
        self.model.load_state_dict(
            torch.load(wave_gan_model_dir / wave_gan_model_ckpt, map_location="cpu")[
                "model"
            ]["generator"]
        )
        self.model.remove_weight_norm()

        # self.use_noise_input = not isinstance(self.model, parallel_wavegan.models.MelGANGenerator)
        self.use_noise_input = True
        self.pad_fn = torch.nn.ReplicationPad1d(
            self.config["generator_params"].get("aux_context_window", 0)
        )
        if self.config["generator_params"]["out_channels"] > 1:
            self.pqmf = PQMF(self.config["generator_params"]["out_channels"])
        else:
            self.pqmf = None

        self.reconstructor = WaveGANReconstruction(
            self.feature_extractor,
            self.model,
            self.pqmf,
            self.use_noise_input,
            self.config,
            self.pad_fn,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply reconstruction defense to input audio."""
        return self.reconstructor(audio)
        # max_len = 8 * 16000
        # audio_len = audio.shape[0]
        # if audio_len <= max_len:
        #     return self.reconstructor(audio)

        # audio = audio[:max_len]
        # return self.reconstructor(audio)

        # logger.info('audio={}'.format(audio.shape))

        # if audio_len <= max_len:
        #     return self.reconstructor(audio)

        # num_chunks = int(math.ceil(audio_len / max_len))
        # chunk_len = audio_len // num_chunks
        # audio_chunks = []
        # t_start = 0
        # for i in range(num_chunks):
        #     t_end = min(t_start + chunk_len, audio_len)
        #     audio_chunks.append(self.reconstructor(audio[t_start:t_end]))
        #     t_start += chunk_len

        # return torch.cat(audio_chunks)


def logmelfilterbank(
    audio: torch.Tensor,
    sampling_rate: int,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: Optional[int] = None,
    window: Optional[Union[str, torch.Tensor]] = "hann",
    num_mels: int = 80,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute log-Mel filterbank features from a waveform tensor.

    Args:
        audio: Audio signal shaped ``(T,)``.
        sampling_rate: Sampling rate in Hz.
        fft_size: FFT size.
        hop_size: Hop size.
        win_length: Window length. If ``None``, uses ``fft_size``.
        window: Window function name (for example ``"hann"``), window
            tensor, or ``None``.
        num_mels: Number of mel filterbank channels.
        fmin: Minimum mel frequency in Hz.
        fmax: Maximum mel frequency in Hz.
        eps: Lower clamp before logarithm for numerical stability.

    Returns:
        Log-Mel tensor shaped ``(num_frames, num_mels)``.
    """
    # # get amplitude spectrogram
    # x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
    #                       win_length=win_length, window=window, pad_mode="reflect")
    # spc = np.abs(x_stft).T  # (#frames, #bins)

    # # get mel basis
    # fmin = 0 if fmin is None else fmin
    # fmax = sampling_rate / 2 if fmax is None else fmax
    # mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    # return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))
    # logger.info('{} {}'.format(audio.shape, audio.device))
    if isinstance(window, str):
        window_name = f"{window}_window"
        if not hasattr(torch, window_name):
            raise ValueError(f"unsupported window type '{window}'")
        win_fn = getattr(torch, window_name)
        cur_win_length = fft_size if win_length is None else win_length
        window = win_fn(cur_win_length, device=audio.device, dtype=audio.dtype)
    elif window is not None:
        window = window.to(device=audio.device, dtype=audio.dtype)

    x_stft2 = (
        torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            pad_mode="reflect",
        ).transpose(0, 1)
        ** 2
    )
    # logger.info('{} {}'.format(x_stft2.shape, x_stft2.device))
    spc = (x_stft2[:, :, 0] + x_stft2[:, :, 1]).sqrt()

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = torch.tensor(
        librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax),
        device=spc.device,
    ).transpose(0, 1)
    return torch.matmul(spc, mel_basis).clamp(min=eps).log10()


class WaveGANFeatureExtractor(nn.Module):
    """Extract normalized log-Mel features for WaveGAN reconstruction."""

    def __init__(self, wave_gan_model_dir: Path) -> None:
        """Load feature-extraction config and normalization statistics.

        Args:
            wave_gan_model_dir: Directory containing ``config.yml`` and
                ``stats.h5``/``stats.npy``.
        """
        super().__init__()
        with open(wave_gan_model_dir / "config.yml") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        win_len = (
            self.config["fft_size"]
            if self.config["win_length"] is None
            else self.config["win_length"]
        )
        self.register_buffer("window", torch.hann_window(win_len))

        # Restore scaler
        if self.config["format"] == "hdf5":
            stats_path = str(wave_gan_model_dir / "stats.h5")
            scaler_mean = read_hdf5(stats_path, "mean")
            scaler_scale = read_hdf5(stats_path, "scale")
        elif self.config["format"] == "npy":
            stats_path = str(wave_gan_model_dir / "stats.npy")
            scaler_mean = np.load(stats_path)[0]
            scaler_scale = np.load(stats_path)[1]
        else:
            raise ValueError("support only hdf5 or npy format.")

        self.register_buffer("scaler_mean", torch.tensor(scaler_mean))
        self.register_buffer("scaler_scale", torch.tensor(scaler_scale))

    def transform(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute normalized log-Mel features from audio.

        Args:
            audio: Audio signal shaped ``(T,)``.

        Returns:
            Normalized log-Mel tensor shaped ``(num_frames, num_mels)``.
        """

        mel = logmelfilterbank(
            audio,
            sampling_rate=self.config["sampling_rate"],
            hop_size=self.config["hop_size"],
            fft_size=self.config["fft_size"],
            win_length=self.config["win_length"],
            window=self.window,
            num_mels=self.config["num_mels"],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
        )

        # Normalize the mel spectrogram
        mel = (mel - self.scaler_mean) / self.scaler_scale

        return mel
