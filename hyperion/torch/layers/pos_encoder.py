"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import Optional, Union

import torch
from torch import nn

from .activation_factory import ActivationFactory as AF


class PosEncoderBase(nn.Module):
    pass


class PosEncoder(PosEncoderBase):
    """Positional encoding.

    Attributes:
      num_feats: embedding dim
      dropout_rate: dropout rate
    """

    def __init__(self, num_feats: int, dropout_rate: float = 0):
        super().__init__()
        self.num_feats = num_feats
        self.dropout_rate = dropout_rate
        self.xscale = math.sqrt(self.num_feats)
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(num_feats={}, dropout_rate={})".format(
            self.__class__.__name__, self.num_feats, self.dropout_rate
        )
        return s

    def _pe(self, x, relative=False):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return self.pe

        pe = torch.zeros(x.size(1), self.num_feats)
        if relative:
            # this is for relative positional encoders
            position = torch.arange(
                x.size(1) - 1, -1, -1, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.num_feats, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.num_feats)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        return self.pe

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x: Input with shape=(batch, time, C)

        Returns:
            x-scaled + pos-encoder
        """
        pe = self._pe(x)
        x = x * self.xscale + pe[:, : x.size(1)]
        if self.dropout_rate > 0:
            return self.dropout(x)
        return x


class RelPosEncoder(PosEncoder):
    """Relative Positional encoding as defined in
       https://arxiv.org/pdf/1901.02860.pdf

       It returns the input and the positional encoder separtely
       so they are mixed in the attention block later.

    Attributes:
      num_feats: embedding dim
      dropout_rate: dropout rate
    """

    def __init__(self, num_feats: int, dropout_rate: float = 0):
        super().__init__(num_feats, dropout_rate)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x: Input with shape=(batch, time, C)

        Returns:
            x-scaled, pos-encoding
        """

        pe = self._pe(x, relative=True)
        x = x * self.xscale
        # we want embedding  [R_L,..., R_0]
        # while in non relative we want [R_0, ..., R_L]
        pos_emb = self.pe[:, -x.size(1) :]
        # this pos_emb is matrix Q in
        # https://arxiv.org/pdf/1901.02860.pdf Appendix B
        # I think it should have been denoted as R,
        # probably a typo in the paper
        if self.dropout_rate > 0:
            x = self.dropout(x)
            pos_emb = self.dropout(pos_emb)

        return x, pos_emb


class NoPosEncoder(PosEncoderBase):
    """This is a dummy class for the case where we
    deactivate the positional encoder

    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Identity map

        Args:
            x: Input with shape=(batch, time, C)

        Returns:
            x
        """
        return x


class ConvPosEncoder(PosEncoderBase):
    """Convolutional positional encoder like the one used in wav2vec2

    Attributes:
      num_feats: number of input/output features
      kernel_size: kernel size of convolution
      num_groups: number of groups of the convolution
      activation: hidden activation
    """

    def __init__(
        self,
        num_feats: int,
        kernel_size: int,
        num_groups: int,
        activation: Union[str, nn.Module],
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            num_feats,
            num_feats,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        self.activation = AF.create(activation)
        self.num_pad_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.num_pad_remove > 0:
            x = x[:, :, : -self.num_pad_remove]

        x = self.activation(x).transpose(1, 2)

        return x


class RotaryPosEncoder(PosEncoderBase):
    """Rotary positiaonal encoder like the one used in LLAMA

    Attributes:

    """

    def __init__(
        self,
        theta: float = 500000,
        scale_freqs: bool = True,
        update_max_seq_length: bool = True,
        original_max_seq_length: Optional[int] = None,
        scaling_factor: float = 8,
        low_freq_factor: float = 1,
        high_freq_factor: float = 4,
    ):
        super().__init__()
        self.theta = theta
        self.freqs_cis = None
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.scaling_factor = scaling_factor
        self.scale_freqs = scale_freqs
        self.update_max_seq_length = update_max_seq_length
        if original_max_seq_length is None:
            original_max_seq_length = 0

        self.register_buffer(
            "max_seq_length",
            torch.as_tensor(original_max_seq_length, dtype=torch.int64),
        )

    @torch.no_grad
    def _scale_freqs(self, freqs: torch.Tensor):

        if self.training and self.update_max_seq_length:
            # if we are updating the max seq length, we don't do scaling
            # since we are just growing the max_seq_length
            # we just scale in inference or if we are training keeping
            # the max seq length fixed.
            return freqs

        max_seq_length = self.max_seq_length
        low_freq_wavelength = max_seq_length / self.low_freq_factor
        high_freq_wavelength = max_seq_length / self.high_freq_factor

        wavelength = 2 * math.pi / freqs

        # wavelength < high_freq_wavelength: do nothing
        # wavelength > low_freq_wavelength: divide by factor
        scaled_freqs = torch.where(
            wavelength > low_freq_wavelength, freqs / self.scaling_factor, freqs
        )
        smooth_factor = (max_seq_length / wavelength - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smoothed_freqs = (
            1 - smooth_factor
        ) * scaled_freqs / self.scaling_factor + smooth_factor * scaled_freqs
        is_medium_freq = ~(wavelength < high_freq_wavelength) * ~(
            wavelength > low_freq_wavelength
        )
        scaled_freqs = torch.where(is_medium_freq, smoothed_freqs, scaled_freqs)
        return scaled_freqs

    #     high_freq_wavelen = old_context_len / high_freq_factor

    #     wavelen = 2 * math.pi / inv_freq
    #     # wavelen < high_freq_wavelen: do nothing
    #     # wavelen > low_freq_wavelen: divide by factor
    #     inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    #     # otherwise: interpolate between the two, using a smooth factor
    #     smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    #     smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    #     is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    #     inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    @torch.no_grad
    def _compute_freqs_cis(self, x: torch.Tensor, start_pos: int):
        length = x.size(1) + start_pos
        if self.freqs_cis is not None:
            freq_length = self.freq_cis.shape[1]
            if length <= freq_length:
                return self.freqs_cis[start_pos:length]

        if (
            self.training
            and self.update_max_seq_length
            and length > self.max_seq_length
        ):
            self.max_seq_length += length - self.max_seq_length

        num_feats = x.size(-1)
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, num_feats, 2, device=x.device)[: num_feats // 2].float()
                / num_feats
            )
        )
        if self.scale_freqs:
            freqs = self._scale_freqs(freqs)

        t = torch.arange(length, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        shape = [d if i == 1 or i == x.dim() - 1 else 1 for i, d in enumerate(x.shape)]
        self.freqs_cis = freqs_cis.view(*shape)
        return freqs_cis[start_pos:length]

    # def apply_rotary_emb(
    #     xq: torch.Tensor,
    #     xk: torch.Tensor,
    #     freqs_cis: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    #     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    #     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    #     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    #     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    #     return xq_out.type_as(xq), xk_out.type_as(xk)

    # def forward(self, query:torch.Tensor, key:torch.Tensor):
    #     query_out = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    #     key_out = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))

    #     freqs_cis = self._compute_freqs_cis(query_out)
    #     query_out = torch.view_as_real(query_out * freqs_cis).flatten(3)
    #     if query.shape[1] != key.shape[1]:
    #         freqs_cis = self._compute_freqs_cis(key_out)

    #     key_out = torch.view_as_real(key_out * freqs_cis).flatten(3)
    #     return query_out.type_as(query), key_out.type_as(key)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        x_out = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = self._compute_freqs_cis(x_out, start_pos)
        x_out = torch.view_as_real(x_out * freqs_cis).flatten(3)
        return x_out.type_as(x)


# def _compute_default_rope_parameters(
#     config: Optional[PretrainedConfig] = None,
#     device: Optional["torch.device"] = None,
#     seq_len: Optional[int] = None,
#     **rope_kwargs,
# ) -> Tuple["torch.Tensor", float]:
#     """
#     Computes the inverse frequencies according to the original RoPE implementation
#     Args:
#         config ([`~transformers.PretrainedConfig`]):
#             The model configuration.
#         device (`torch.device`):
#             The device to use for initialization of the inverse frequencies.
#         seq_len (`int`, *optional*):
#             The current sequence length. Unused for this type of RoPE.
#         rope_kwargs (`Dict`, *optional*):
#             BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
#     Returns:
#         Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
#         post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
#     """
#     if config is not None and len(rope_kwargs) > 0:
#         raise ValueError(
#             "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
#             f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
#         )
#     if len(rope_kwargs) > 0:
#         base = rope_kwargs["base"]
#         dim = rope_kwargs["dim"]
#     elif config is not None:
#         base = config.rope_theta
#         partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
#         head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         dim = int(head_dim * partial_rotary_factor)

#     attention_factor = 1.0  # Unused in this type of RoPE

#     # Compute the inverse frequencies
#     inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
#     return inv_freq, attention_factor


# def _compute_llama3_parameters(
#     config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
# ) -> Tuple["torch.Tensor", float]:
#     """
#     Computes the inverse frequencies for llama 3.1.

#     Args:
#         config ([`~transformers.PretrainedConfig`]):
#             The model configuration.
#         device (`torch.device`):
#             The device to use for initialization of the inverse frequencies.
#         seq_len (`int`, *optional*):
#             The current sequence length. Unused for this type of RoPE.
#         rope_kwargs (`Dict`, *optional*):
#             BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
#     Returns:
#         Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
#         post-processing scaling factor applied to the computed cos/sin.
#     """
#     # Gets the default RoPE parameters
#     inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

#     factor = config.rope_scaling["factor"]  # `8` in the original implementation
#     low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
#     high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
#     old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

#     low_freq_wavelen = old_context_len / low_freq_factor
#     high_freq_wavelen = old_context_len / high_freq_factor

#     wavelen = 2 * math.pi / inv_freq
#     # wavelen < high_freq_wavelen: do nothing
#     # wavelen > low_freq_wavelen: divide by factor
#     inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
#     # otherwise: interpolate between the two, using a smooth factor
#     smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
#     smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
#     is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
#     inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

#     return inv_freq_llama, attention_factor

# def _check_received_keys(rope_type: str, received_keys: set, required_keys: set, optional_keys: Optional[set] = None):
#     """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
#     # BC: "rope_type" was originally "type" -- let's gracefully handle it
#     if "rope_type" not in received_keys and "type" in received_keys:
#         received_keys -= {"type"}
#         received_keys.add("rope_type")

#     missing_keys = required_keys - received_keys
#     if missing_keys:
#         raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

#     if optional_keys is not None:
#         unused_keys = received_keys - required_keys - optional_keys
#     else:
#         unused_keys = received_keys - required_keys
#     if unused_keys:
#         logger.warning(f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}")

# def _validate_llama3_parameters(config: PretrainedConfig):
#     rope_scaling = config.rope_scaling
#     rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
#     required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
#     received_keys = set(rope_scaling.keys())
#     _check_received_keys(rope_type, received_keys, required_keys)

#     factor = rope_scaling["factor"]
#     if factor is None or not isinstance(factor, float) or factor < 1.0:
#         logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

#     low_freq_factor = rope_scaling["low_freq_factor"]
#     high_freq_factor = rope_scaling["high_freq_factor"]
#     if low_freq_factor is None or not isinstance(low_freq_factor, float):
#         logger.warning(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
#     if high_freq_factor is None or not isinstance(high_freq_factor, float):
#         logger.warning(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
#     if high_freq_factor <= low_freq_factor:
#         logger.warning(
#             "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
#             f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
#         )

#     original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
#     if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
#         logger.warning(
#             "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
#             f"{original_max_position_embeddings}"
#         )
#     if original_max_position_embeddings >= config.max_position_embeddings:
#         logger.warning(
#             "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
#             f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
#         )
