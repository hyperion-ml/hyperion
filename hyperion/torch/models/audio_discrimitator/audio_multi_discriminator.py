"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils import HyperDataClass
from ...hyper_torch_model import HyperTorchModel
from ...layers import ActivationFactory as AF


class AudioDiscriminatorTrainMode(str, Enum):
    """Training modes supported by the audio discriminator stack.

    Attributes:
        FULL: Train the discriminator normally.
        FROZEN: Keep the discriminator frozen.
    """

    FULL = "full"
    FROZEN = "frozen"

    @staticmethod
    def choices() -> List[str]:
        """Return the valid enum values.

        Returns:
            The list of supported training-mode string values.
        """
        return [o.value for o in AudioDiscriminatorTrainMode]


@dataclass
class AudioMultiDiscriminatorOutput(HyperDataClass):
    """Container for the multi-discriminator outputs.

    Attributes:
        msd_outputs: Outputs from the multi-scale discriminators.
        mpd_outputs: Outputs from the multi-period discriminators.
        mrsp_outputs: Outputs from the multi-resolution spectrogram discriminators.
        msd_fmaps: Feature maps from the multi-scale discriminators.
        mpd_fmaps: Feature maps from the multi-period discriminators.
        mrsp_fmaps: Feature maps from the multi-resolution spectrogram discriminators.
    """

    msd_outputs: List[torch.Tensor]
    mpd_outputs: List[torch.Tensor]
    mrsp_outputs: List[torch.Tensor]
    msd_fmaps: List[List[torch.Tensor]]
    mpd_fmaps: List[List[torch.Tensor]]
    mrsp_fmaps: List[List[torch.Tensor]]


MRSD_BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class _NormConv2d(nn.Module):
    """2D convolution wrapped with a normalization parametrization.

    Attributes:
        conv: Normalized convolution module.
        activation: Activation module applied after convolution.
    """

    def __init__(
        self, norm: Any, activation: Optional[nn.Module], *args: Any, **kwargs: Any
    ) -> None:
        """Wrap a 2D convolution with a normalization parametrization.

        Args:
            norm: Normalization wrapper applied to the convolution module.
            activation: Optional activation module applied after the convolution.
            *args: Positional arguments forwarded to :class:`torch.nn.Conv2d`.
            **kwargs: Keyword arguments forwarded to :class:`torch.nn.Conv2d`.
        """
        super().__init__()
        self.conv = norm(nn.Conv2d(*args, **kwargs))
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalized convolution and activation.

        Args:
            x: Input tensor.

        Returns:
            The transformed tensor.
        """
        return self.activation(self.conv(x))


class _NormConv1d(nn.Module):
    """1D convolution wrapped with a normalization parametrization.

    Attributes:
        conv: Normalized convolution module.
        activation: Activation module applied after convolution.
    """

    def __init__(
        self, norm: Any, activation: Optional[nn.Module], *args: Any, **kwargs: Any
    ) -> None:
        """Wrap a 1D convolution with a normalization parametrization.

        Args:
            norm: Normalization wrapper applied to the convolution module.
            activation: Optional activation module applied after the convolution.
            *args: Positional arguments forwarded to :class:`torch.nn.Conv1d`.
            **kwargs: Keyword arguments forwarded to :class:`torch.nn.Conv1d`.
        """
        super().__init__()
        self.conv = norm(nn.Conv1d(*args, **kwargs))
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalized convolution and activation.

        Args:
            x: Input tensor.

        Returns:
            The transformed tensor.
        """
        return self.activation(self.conv(x))


class AudioPeriodDiscriminator(HyperTorchModel):
    """
    A discriminator designed to capture periodic structures in audio waveforms,
    such as pitch harmonics, by reshaping the waveform into 2D blocks of a given period.

    Attributes:
        period (int): Number of samples per period to reshape input.
        kernel_size (int): Kernel size of internal convolutions.
        stride (int): Stride used in internal convolutions.
        out_kernel_size (int): Kernel size for the output layer.
        activation (str): Activation function name ('leakyrelu', etc.).
        use_spectral_norm (bool): Whether to use spectral normalization (if False, uses weight norm).
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        out_kernel_size: int = 3,
        activation: Union[str, nn.Module] = "leakyrelu",
        use_spectral_norm: bool = False,
    ) -> None:
        """Build a period-based audio discriminator.

        Args:
            period: Number of samples used to reshape each waveform period.
            kernel_size: Kernel size for the internal convolution blocks.
            stride: Stride for the internal convolution blocks.
            out_kernel_size: Kernel size of the final output convolution.
            activation: Activation name or module to apply after convolutions.
            use_spectral_norm: If ``True``, use spectral normalization instead of weight norm.
        """
        super().__init__()
        self.period = period
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_spectral_norm = use_spectral_norm
        self.activation = activation
        self.out_kernel_size = out_kernel_size

        norm_f = (
            nn.utils.parametrizations.weight_norm
            if not use_spectral_norm
            else nn.utils.parametrizations.spectral_norm
        )
        if activation == "leakyrelu":
            act_f = nn.LeakyReLU(negative_slope=0.1)
        else:
            act_f = AF.create(activation)

        padding = (int((kernel_size - 1) // 2), 0)
        kernel_size_2d = (kernel_size, 1)
        stride_2d = (stride, 1)
        self.layers = nn.ModuleList(
            [
                _NormConv2d(
                    norm_f, act_f, 1, 32, kernel_size_2d, stride_2d, padding=padding
                ),
                _NormConv2d(
                    norm_f, act_f, 32, 128, kernel_size_2d, stride_2d, padding=padding
                ),
                _NormConv2d(
                    norm_f, act_f, 128, 512, kernel_size_2d, stride_2d, padding=padding
                ),
                _NormConv2d(
                    norm_f, act_f, 512, 1024, kernel_size_2d, stride_2d, padding=padding
                ),
                _NormConv2d(
                    norm_f, act_f, 1024, 1024, kernel_size_2d, 1, padding=padding
                ),
            ]
        )
        kernel_size_2d = (out_kernel_size, 1)
        padding = (int((out_kernel_size - 1) // 2), 0)
        self.out_layer = _NormConv2d(
            norm_f, None, 1024, 1, kernel_size=kernel_size_2d, padding=padding
        )

    def pad_to_period(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the waveform length to an integer multiple of the period.

        Args:
            x: Input waveform tensor of shape ``(B, C, T)``.

        Returns:
            The padded waveform tensor.
        """
        t = x.shape[-1]
        if t % self.period == 0:
            return x
        x = nn.functional.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Score the input waveform and collect intermediate feature maps.

        Args:
            x: Input waveform tensor of shape ``(B, C, T)``.

        Returns:
            A tuple with the flattened discriminator output and the feature maps.
        """
        fmaps = []
        x = self.pad_to_period(x)
        x = x.reshape(x.shape[0], x.shape[1], -1, self.period)  # (b, c, l, p)

        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)

        x = self.out_layer(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmaps

    def remove_weight_norm(self) -> None:
        """Remove weight-normalization parametrizations from all convolutions."""
        for layer in self.layers:
            nn.utils.parametrize.remove_parametrizations(
                layer.conv, "weight", leave_parametrized=False
            )
        nn.utils.parametrize.remove_parametrizations(
            self.out_layer.conv, "weight", leave_parametrized=False
        )


class AudioScaleDiscriminator(HyperTorchModel):
    """
    A 1D convolutional discriminator that operates at different input scales
    (e.g., full, 2×, 4× downsampled audio) to capture multi-resolution time-domain patterns.

    Attributes:
        scale (int): Downsampling factor for the input audio.
        kernel_sizes (List[int]): Convolution kernel sizes for layers.
        strides (List[int]): Strides used in each convolutional layer.
        out_kernel_size (int): Kernel size for the final output convolution.
        activation (str): Activation function to use ('leakyrelu', etc.).
        use_spectral_norm (bool): Whether to use spectral normalization (else uses weight norm).
    """

    def __init__(
        self,
        scale: int = 1,
        kernel_sizes: List[int] = [15, 41, 5],
        strides: List[int] = [1, 4, 1],
        out_kernel_size: int = 3,
        activation: Union[str, nn.Module] = "leakyrelu",
        use_spectral_norm: bool = False,
    ) -> None:
        """Build a multi-scale waveform discriminator.

        Args:
            scale: Optional input downsampling factor before the convolutions.
            kernel_sizes: Convolution kernel sizes for the internal blocks.
            strides: Convolution strides for the internal blocks.
            out_kernel_size: Kernel size of the final output convolution.
            activation: Activation name or module to apply after convolutions.
            use_spectral_norm: If ``True``, use spectral normalization instead of weight norm.
        """
        super().__init__()
        self.scale = scale
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.use_spectral_norm = use_spectral_norm
        self.activation = activation
        self.out_kernel_size = out_kernel_size

        norm_f = (
            nn.utils.parametrizations.weight_norm
            if not use_spectral_norm
            else nn.utils.parametrizations.spectral_norm
        )
        if activation == "leakyrelu":
            act_f = nn.LeakyReLU(negative_slope=0.1)
        else:
            act_f = AF.create(activation)

        if scale > 1:
            self.pooling = nn.AvgPool1d(
                kernel_size=2 * scale, stride=scale, padding=scale
            )

        paddings = [(k - 1) // 2 for k in kernel_sizes]
        self.layers = nn.ModuleList(
            [
                _NormConv1d(
                    norm_f,
                    act_f,
                    1,
                    16,
                    kernel_sizes[0],
                    strides[0],
                    groups=1,
                    padding=paddings[0],
                ),
                _NormConv1d(
                    norm_f,
                    act_f,
                    16,
                    64,
                    kernel_sizes[1],
                    strides[1],
                    groups=4,
                    padding=paddings[1],
                ),
                _NormConv1d(
                    norm_f,
                    act_f,
                    64,
                    256,
                    kernel_sizes[1],
                    strides[1],
                    groups=16,
                    padding=paddings[1],
                ),
                _NormConv1d(
                    norm_f,
                    act_f,
                    256,
                    1024,
                    kernel_sizes[1],
                    strides[1],
                    groups=64,
                    padding=paddings[1],
                ),
                _NormConv1d(
                    norm_f,
                    act_f,
                    1024,
                    1024,
                    kernel_sizes[1],
                    strides[1],
                    groups=256,
                    padding=paddings[1],
                ),
                _NormConv1d(
                    norm_f,
                    act_f,
                    1024,
                    1024,
                    kernel_sizes[2],
                    strides[2],
                    groups=1,
                    padding=paddings[2],
                ),
            ]
        )

        padding = (out_kernel_size - 1) // 2
        self.out_layer = _NormConv1d(
            norm_f, None, 1024, 1, kernel_size=out_kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Score the input waveform and collect intermediate feature maps.

        Args:
            x: Input waveform tensor of shape ``(B, C, T)``.

        Returns:
            A tuple with the flattened discriminator output and the feature maps.
        """
        fmaps = []
        if self.scale > 1:
            x = self.pooling(x)

        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)

        x = self.out_layer(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmaps


class AudioSpectrogramDiscriminator(HyperTorchModel):
    """
    A spectral discriminator that analyzes audio in multiple frequency bands of the STFT,
    applying a stack of 2D convolutions to each band individually. Designed to capture
    detailed time-frequency patterns.

    Attributes:
        window_length (int): STFT window size.
        hop_length (int): STFT hop size.
        freq_kernel_sizes (List[int]): Kernel sizes along the frequency axis.
        time_kernel_sizes (List[int]): Kernel sizes along the time axis.
        strides (List[int]): Strides used in each conv block.
        out_kernel_size (int): Final output layer kernel size.
        bands (List[Tuple[float, float]]): Frequency bands as fractions of Nyquist (e.g., (0.0, 0.1)).
        activation (str): Activation function to use.
        use_spectral_norm (bool): Whether to use spectral normalization in conv layers.
    """

    def __init__(
        self,
        window_length: int = 1024,
        hop_length: int = 256,
        freq_kernel_sizes: List[int] = [9, 9, 3],
        time_kernel_sizes: List[int] = [3, 3, 3],
        strides: List[int] = [1, 2, 1],
        out_kernel_size: int = 3,
        bands: List[Tuple[float, float]] = MRSD_BANDS,
        activation: Union[str, nn.Module] = "leakyrelu",
        use_spectral_norm: bool = False,
    ) -> None:
        """Build a multi-resolution spectrogram discriminator.

        Args:
            window_length: STFT window length used to compute the spectrogram.
            hop_length: STFT hop length used to compute the spectrogram.
            freq_kernel_sizes: Convolution kernel sizes along the frequency axis.
            time_kernel_sizes: Convolution kernel sizes along the time axis.
            strides: Convolution strides for the stacked conv blocks.
            out_kernel_size: Kernel size of the final output convolution.
            bands: Frequency bands to analyze, normalized to the Nyquist range.
            activation: Activation name or module to apply after convolutions.
            use_spectral_norm: If ``True``, use spectral normalization instead of weight norm.
        """
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.freq_kernel_sizes = freq_kernel_sizes
        self.time_kernel_sizes = time_kernel_sizes
        self.strides = strides
        self.use_spectral_norm = use_spectral_norm
        self.activation = activation
        self.out_kernel_size = out_kernel_size
        num_fft = window_length // 2 + 1
        bands = [(int(b[0] * num_fft), int(b[1] * num_fft)) for b in bands]
        self.bands = bands
        norm_f = (
            nn.utils.parametrizations.weight_norm
            if not use_spectral_norm
            else nn.utils.parametrizations.spectral_norm
        )
        if activation == "leakyrelu":
            act_f = nn.LeakyReLU(negative_slope=0.1)
        else:
            act_f = AF.create(activation)

        paddings = [
            ((tk - 1) // 2, (fk - 1) // 2)
            for tk, fk in zip(time_kernel_sizes, freq_kernel_sizes)
        ]

        channels = 32
        layers = []
        for _ in range(len(self.bands)):
            band_layer = nn.ModuleList(
                [
                    _NormConv2d(
                        norm_f,
                        act_f,
                        2,
                        channels,
                        (time_kernel_sizes[0], freq_kernel_sizes[0]),
                        (1, strides[0]),
                        padding=paddings[0],
                    ),
                    _NormConv2d(
                        norm_f,
                        act_f,
                        channels,
                        channels,
                        (time_kernel_sizes[1], freq_kernel_sizes[1]),
                        (1, strides[1]),
                        padding=paddings[1],
                    ),
                    _NormConv2d(
                        norm_f,
                        act_f,
                        channels,
                        channels,
                        (time_kernel_sizes[1], freq_kernel_sizes[1]),
                        (1, strides[1]),
                        padding=paddings[1],
                    ),
                    _NormConv2d(
                        norm_f,
                        act_f,
                        channels,
                        channels,
                        (time_kernel_sizes[1], freq_kernel_sizes[1]),
                        (1, strides[1]),
                        padding=paddings[1],
                    ),
                    _NormConv2d(
                        norm_f,
                        act_f,
                        channels,
                        channels,
                        (time_kernel_sizes[2], freq_kernel_sizes[2]),
                        (1, strides[2]),
                        padding=paddings[2],
                    ),
                ]
            )
            layers.append(band_layer)

        self.layers = nn.ModuleList(layers)

        padding = (out_kernel_size - 1) // 2
        self.out_layer = _NormConv2d(
            norm_f,
            None,
            channels,
            1,
            kernel_size=(out_kernel_size, out_kernel_size),
            padding=(padding, padding),
        )

        window = torch.hann_window(window_length)
        self.register_buffer("window", window)

    def get_spectrogram_bands(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute the band-limited STFT views used by the discriminator.

        Args:
            x: Input waveform tensor of shape ``(B, C, T)`` or ``(B, T)``.

        Returns:
            A list of spectrogram tensors, one per configured frequency band.
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) -> (B, T)

        spec = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            return_complex=True,
        )
        # Convert to real view manually if needed (to match old shape)
        # spec: (B, F, T) complex -> (B, F, T, 2) real
        spec = torch.view_as_real(spec)  # shape: (B, F, T, 2)
        # (B, F, T, 2) -> (B, 2, T, F)
        spec = spec.permute(0, 3, 2, 1).contiguous()
        spec_bands = [spec[..., b[0] : b[1]] for b in self.bands]
        return spec_bands

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Score each spectrogram band and collect intermediate feature maps.

        Args:
            x: Input waveform tensor of shape ``(B, C, T)`` or ``(B, T)``.

        Returns:
            A tuple with the flattened discriminator output and the feature maps.
        """
        x_bands = self.get_spectrogram_bands(x)
        fmaps = []

        x = []
        for x_band, band_layer in zip(x_bands, self.layers):
            for band_layer in band_layer:
                x_band = band_layer(x_band)
                fmaps.append(x_band)

            x.append(x_band)

        x = torch.cat(x, dim=-1)
        x = self.out_layer(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps

    def remove_weight_norm(self) -> None:
        """Remove weight-normalization parametrizations from all convolutions."""
        for layer_group in self.layers:
            for layer in layer_group:
                nn.utils.parametrize.remove_parametrizations(
                    layer.conv, "weight", leave_parametrized=False
                )
        nn.utils.parametrize.remove_parametrizations(
            self.out_layer.conv, "weight", leave_parametrized=False
        )


class AudioMultiDiscriminator(HyperTorchModel):
    """
    A composite adversarial discriminator composed of multiple sub-discriminators:
    - Multi-Scale (MSD): Analyzes waveform at different temporal scales.
    - Multi-Period (MPD): Analyzes reshaped waveforms by periodicity.
    - Multi-Resolution Spectrogram (MRSD): Analyzes STFTs across multiple bands and resolutions.

    Args:
        use_msd (bool): Whether to use multi-scale waveform discriminators.
        use_mpd (bool): Whether to use multi-period waveform discriminators.
        use_mrsd (bool): Whether to use multi-resolution spectral discriminators.
        scales (List[int]): Temporal downsampling scales for MSD.
        msd_kernel_sizes (List[int]): Kernel sizes for MSD layers.
        msd_strides (List[int]): Strides for MSD layers.
        msd_out_kernel_size (int): Output conv kernel size for MSD.
        periods (List[int]): Period values for reshaping input in MPD.
        mpd_kernel_size (int): Kernel size for MPD conv layers.
        mpd_stride (int): Stride for MPD conv layers.
        mpd_out_kernel_size (int): Output conv kernel size for MPD.
        mrsp_win_sizes (List[int]): STFT window lengths for MRSD.
        mrsp_hop_sizes (List[int]): STFT hop sizes for MRSD.
        mrsp_freq_kernel_sizes (List[int]): Frequency axis conv kernel sizes in MRSD.
        mrsp_time_kernel_sizes (List[int]): Time axis conv kernel sizes in MRSD.
        mrsp_strides (List[int]): Strides for MRSD convolutional layers.
        out_kernel_size (int): Final output layer kernel size.
        mrsp_bands (List[Tuple[float, float]]): Frequency bands for MRSD (normalized 0–1).
        activation (str): Activation function name.
        use_spectral_norm (bool): Whether to use spectral normalization instead of weight norm.

    Attributes:
        discriminators: Module list containing the enabled sub-discriminators.
        use_msd: Whether multi-scale discriminators are enabled.
        use_mpd: Whether multi-period discriminators are enabled.
        use_mrsd: Whether multi-resolution spectrogram discriminators are enabled.
        scales: Temporal scales used by the multi-scale discriminators.
        periods: Periods used by the multi-period discriminators.
        mrsp_win_sizes: STFT window sizes used by the spectrogram discriminators.
    """

    def __init__(
        self,
        use_msd: bool = True,
        use_mpd: bool = True,
        use_mrsd: bool = True,
        scales: List[int] = [1],
        msd_kernel_sizes: List[int] = [15, 41, 5],
        msd_strides: List[int] = [1, 4, 1],
        msd_out_kernel_size: int = 3,
        periods: List[int] = [2, 3, 5, 7, 11],
        mpd_kernel_size: int = 5,
        mpd_stride: int = 3,
        mpd_out_kernel_size: int = 3,
        mrsp_win_sizes: List[int] = [2048, 1024, 512],
        mrsp_hop_sizes: List[int] = [512, 256, 128],
        mrsp_freq_kernel_sizes: List[int] = [9, 9, 3],
        mrsp_time_kernel_sizes: List[int] = [3, 3, 3],
        mrsp_strides: List[int] = [1, 2, 1],
        out_kernel_size: int = 3,
        mrsp_bands: List[Union[Tuple[float, float], float]] = MRSD_BANDS,
        activation: Union[str, nn.Module] = "leakyrelu",
        use_spectral_norm: bool = False,
    ) -> None:
        """Build the composite audio discriminator.

        Args:
            use_msd: Whether to instantiate multi-scale waveform discriminators.
            use_mpd: Whether to instantiate multi-period waveform discriminators.
            use_mrsd: Whether to instantiate multi-resolution spectrogram discriminators.
            scales: Temporal downsampling scales for MSD.
            msd_kernel_sizes: Kernel sizes for MSD layers.
            msd_strides: Strides for MSD layers.
            msd_out_kernel_size: Output convolution kernel size for MSD.
            periods: Period values for MPD.
            mpd_kernel_size: Kernel size for MPD conv layers.
            mpd_stride: Stride for MPD conv layers.
            mpd_out_kernel_size: Output convolution kernel size for MPD.
            mrsp_win_sizes: STFT window lengths for MRSD.
            mrsp_hop_sizes: STFT hop sizes for MRSD.
            mrsp_freq_kernel_sizes: Frequency-axis kernel sizes for MRSD.
            mrsp_time_kernel_sizes: Time-axis kernel sizes for MRSD.
            mrsp_strides: Convolution strides for MRSD.
            out_kernel_size: Final output convolution kernel size.
            mrsp_bands: Frequency bands for MRSD, either as tuples or flattened pairs.
            activation: Activation name or module to apply after convolutions.
            use_spectral_norm: If ``True``, use spectral normalization instead of weight norm.
        """
        super().__init__()
        if len(mrsp_bands) > 0 and not isinstance(mrsp_bands[0], tuple):
            if len(mrsp_bands) % 2 != 0:
                raise ValueError(
                    "mrsp_bands must contain an even number of values when provided as a flattened list"
                )
            mrsp_bands = list(zip(mrsp_bands[::2], mrsp_bands[1::2]))

        self.use_msd = use_msd
        self.use_mpd = use_mpd
        self.use_mrsd = use_mrsd
        self.scales = scales
        self.periods = periods
        self.msd_kernel_sizes = msd_kernel_sizes
        self.msd_strides = msd_strides
        self.msd_out_kernel_size = msd_out_kernel_size
        self.mpd_kernel_size = mpd_kernel_size
        self.mpd_stride = mpd_stride
        self.mpd_out_kernel_size = mpd_out_kernel_size
        self.mrsp_win_sizes = mrsp_win_sizes
        self.mrsp_hop_sizes = mrsp_hop_sizes
        self.mrsp_freq_kernel_sizes = mrsp_freq_kernel_sizes
        self.mrsp_time_kernel_sizes = mrsp_time_kernel_sizes
        self.mrsp_strides = mrsp_strides
        self.out_kernel_size = out_kernel_size
        self.mrsp_bands = mrsp_bands
        self.activation = activation
        self.use_spectral_norm = use_spectral_norm

        if not use_msd and not use_mpd and not use_mrsd:
            raise ValueError("At least one discriminator must be used")
        if not isinstance(scales, list):
            raise TypeError("scales must be a list of integers")
        if not isinstance(periods, list):
            raise TypeError("periods must be a list of integers")
        if not isinstance(mrsp_win_sizes, list):
            raise TypeError("mrsp_win_sizes must be a list of integers")
        if not isinstance(mrsp_hop_sizes, list):
            raise TypeError("mrsp_hop_sizes must be a list of integers")
        if not isinstance(mrsp_freq_kernel_sizes, list):
            raise TypeError("mrsp_freq_kernel_sizes must be a list of integers")
        if not isinstance(mrsp_time_kernel_sizes, list):
            raise TypeError("mrsp_time_kernel_sizes must be a list of integers")
        if not isinstance(mrsp_strides, list):
            raise TypeError("mrsp_strides must be a list of integers")
        if not isinstance(mrsp_bands, list):
            raise TypeError("mrsp_bands must be a list of tuples (float, float)")
        if len(mrsp_win_sizes) != len(mrsp_hop_sizes):
            raise ValueError(
                "mrsp_win_sizes and mrsp_hop_sizes must have the same length"
            )
        if len(mrsp_freq_kernel_sizes) != len(mrsp_time_kernel_sizes) or len(
            mrsp_freq_kernel_sizes
        ) != len(mrsp_strides):
            raise ValueError(
                "mrsp_freq_kernel_sizes, mrsp_time_kernel_sizes and mrsp_strides must have the same length"
            )
        if len(mrsp_bands) == 0:
            raise ValueError("mrsp_bands must not be empty")
        if len(scales) == 0 and use_msd:
            raise ValueError("scales must not be empty if use_msd is True")
        if len(periods) == 0 and use_mpd:
            raise ValueError("periods must not be empty if use_mpd is True")
        if len(mrsp_win_sizes) == 0 and use_mrsd:
            raise ValueError("mrsp_win_sizes must not be empty if use_mrsd is True")
        if len(mrsp_hop_sizes) == 0 and use_mrsd:
            raise ValueError("mrsp_hop_sizes must not be empty if use_mrsd is True")
        if len(mrsp_freq_kernel_sizes) == 0 and use_mrsd:
            raise ValueError(
                "mrsp_freq_kernel_sizes must not be empty if use_mrsd is True"
            )
        if len(mrsp_time_kernel_sizes) == 0 and use_mrsd:
            raise ValueError(
                "mrsp_time_kernel_sizes must not be empty if use_mrsd is True"
            )
        if len(mrsp_strides) == 0 and use_mrsd:
            raise ValueError("mrsp_strides must not be empty if use_mrsd is True")
        if len(mrsp_bands) == 0 and use_mrsd:
            raise ValueError("mrsp_bands must not be empty if use_mrsd is True")

        discs = []
        if use_msd:
            discs += [
                AudioScaleDiscriminator(
                    s,
                    kernel_sizes=msd_kernel_sizes,
                    strides=msd_strides,
                    out_kernel_size=msd_out_kernel_size,
                    activation=activation,
                    use_spectral_norm=use_spectral_norm,
                )
                for s in scales
            ]

        if use_mpd:
            discs += [
                AudioPeriodDiscriminator(
                    p,
                    kernel_size=mpd_kernel_size,
                    stride=mpd_stride,
                    out_kernel_size=mpd_out_kernel_size,
                    activation=activation,
                    use_spectral_norm=use_spectral_norm,
                )
                for p in periods
            ]

        if use_mrsd:
            discs += [
                AudioSpectrogramDiscriminator(
                    window_length=w,
                    hop_length=h,
                    freq_kernel_sizes=mrsp_freq_kernel_sizes,
                    time_kernel_sizes=mrsp_time_kernel_sizes,
                    strides=mrsp_strides,
                    out_kernel_size=out_kernel_size,
                    bands=mrsp_bands,
                    activation=activation,
                    use_spectral_norm=use_spectral_norm,
                )
                for w, h in zip(mrsp_win_sizes, mrsp_hop_sizes)
            ]

        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Center and peak-normalize input audio.

        Args:
            x: Input tensor of shape ``(B, C, T)`` or ``(B, T)``.

        Returns:
            The normalized tensor.
        """
        if x.dim() == 2:
            # If input is 2D, assume it is (B, T) and add a channel dimension
            x = x.unsqueeze(1)

        # Remove DC offset
        x = x - x.mean(dim=-1, keepdim=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Run the enabled discriminators on the input waveform.

        Args:
            x: Input waveform tensor of shape ``(B, C, T)`` or ``(B, T)``.

        Returns:
            A tuple containing the discriminator outputs and feature maps.
        """

        x = self.preprocess(x)
        outputs = []
        fmaps = []
        for d in self.discriminators:
            output_i, fmaps_i = d(x)
            outputs.append(output_i)
            fmaps.append(fmaps_i)

        return outputs, fmaps

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable configuration of this module.

        Returns:
            A dictionary with constructor arguments and base class metadata.
        """
        config = {
            "use_msd": self.use_msd,
            "use_mpd": self.use_mpd,
            "use_mrsd": self.use_mrsd,
            "scales": self.scales,
            "msd_kernel_sizes": self.msd_kernel_sizes,
            "msd_strides": self.msd_strides,
            "msd_out_kernel_size": self.msd_out_kernel_size,
            "periods": self.periods,
            "mpd_kernel_size": self.mpd_kernel_size,
            "mpd_stride": self.mpd_stride,
            "mpd_out_kernel_size": self.mpd_out_kernel_size,
            "mrsp_win_sizes": self.mrsp_win_sizes,
            "mrsp_hop_sizes": self.mrsp_hop_sizes,
            "mrsp_freq_kernel_sizes": self.mrsp_freq_kernel_sizes,
            "mrsp_time_kernel_sizes": self.mrsp_time_kernel_sizes,
            "mrsp_strides": self.mrsp_strides,
            "out_kernel_size": self.out_kernel_size,
            "mrsp_bands": self.mrsp_bands,
            "activation": self.activation,
            "use_spectral_norm": self.use_spectral_norm,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set()
    ) -> None:
        """Add constructor arguments to a JSONArgParse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix used for nested parsers.
            skip: Argument names to omit from the parser.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "use_msd" not in skip:
            parser.add_argument(
                "--use-msd",
                default=True,
                action=ActionYesNo,
                help="Whether to use multi-scale waveform discriminators",
            )
        if "use_mpd" not in skip:
            parser.add_argument(
                "--use-mpd",
                default=True,
                action=ActionYesNo,
                help="Whether to use multi-period waveform discriminators",
            )
        if "use_mrsd" not in skip:
            parser.add_argument(
                "--use-mrsd",
                default=True,
                action=ActionYesNo,
                help="Whether to use multi-resolution spectrogram discriminators",
            )
        if "scales" not in skip:
            parser.add_argument(
                "--scales",
                type=int,
                nargs="+",
                default=[1],
                help="Temporal downsampling scales for MSD",
            )
        if "msd_kernel_sizes" not in skip:
            parser.add_argument(
                "--msd-kernel-sizes",
                type=int,
                nargs="+",
                default=[15, 41, 5],
                help="Kernel sizes for MSD layers",
            )
        if "msd_strides" not in skip:
            parser.add_argument(
                "--msd-strides",
                type=int,
                nargs="+",
                default=[1, 4, 1],
                help="Strides for MSD layers",
            )
        if "msd_out_kernel_size" not in skip:
            parser.add_argument(
                "--msd-out-kernel-size",
                type=int,
                default=3,
                help="Output convolution kernel size for MSD",
            )
        if "periods" not in skip:
            parser.add_argument(
                "--periods",
                type=int,
                nargs="+",
                default=[2, 3, 5, 7, 11],
                help="Periods for MPD",
            )
        if "mpd_kernel_size" not in skip:
            parser.add_argument(
                "--mpd-kernel-size",
                type=int,
                default=5,
                help="Kernel size for MPD layers",
            )
        if "mpd_stride" not in skip:
            parser.add_argument(
                "--mpd-stride",
                type=int,
                default=3,
                help="Stride for MPD layers",
            )
        if "mpd_out_kernel_size" not in skip:
            parser.add_argument(
                "--mpd-out-kernel-size",
                type=int,
                default=3,
                help="Output convolution kernel size for MPD",
            )
        if "mrsp_win_sizes" not in skip:
            parser.add_argument(
                "--mrsp-win-sizes",
                type=int,
                nargs="+",
                default=[2048, 1024, 512],
                help="Window sizes for MRSD STFT",
            )
        if "mrsp_hop_sizes" not in skip:
            parser.add_argument(
                "--mrsp-hop-sizes",
                type=int,
                nargs="+",
                default=[512, 256, 128],
                help="Hop sizes for MRSD STFT",
            )
        if "mrsp_freq_kernel_sizes" not in skip:
            parser.add_argument(
                "--mrsp-freq-kernel-sizes",
                type=int,
                nargs="+",
                default=[9, 9, 3],
                help="Frequency kernel sizes for MRSD",
            )
        if "mrsp_time_kernel_sizes" not in skip:
            parser.add_argument(
                "--mrsp-time-kernel-sizes",
                type=int,
                nargs="+",
                default=[3, 3, 3],
                help="Time kernel sizes for MRSD",
            )
        if "mrsp_strides" not in skip:
            parser.add_argument(
                "--mrsp-strides",
                type=int,
                nargs="+",
                default=[1, 2, 1],
                help="Strides for MRSD conv layers",
            )
        if "out_kernel_size" not in skip:
            parser.add_argument(
                "--out-kernel-size",
                type=int,
                default=3,
                help="Final output conv kernel size for MRSD",
            )
        if "mrsp_bands" not in skip:
            parser.add_argument(
                "--mrsp-bands",
                type=float,
                nargs="+",
                default=[v for band in MRSD_BANDS for v in band],
                help="Frequency bands for MRSD, flattened as list",
            )
        if "activation" not in skip:
            parser.add_argument(
                "--activation",
                type=str,
                default="leakyrelu",
                help="Activation function name",
            )
        if "use_spectral_norm" not in skip:
            parser.add_argument(
                "--use-spectral-norm",
                default=False,
                action=ActionYesNo,
                help="Use spectral normalization instead of weight norm",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
