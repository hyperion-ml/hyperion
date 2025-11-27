"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingCausalConv1d(nn.Conv1d):
    """
    Causal Conv1d with explicit padding and a streaming API.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Stride (padding fixed to 0 in this wrapper).
        dilation: Dilation factor.
        groups: Group count.
        weight: Convolution weights.
        bias: Optional bias.
        look_ahead: Look-ahead samples available at stream time.
        receptive_left_total: Total left receptive field (k-1)*d of the kernel.
        receptive_left_effective: Past required when using look-ahead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,  # ignored; forced to 0
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        look_ahead: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            stride: Stride (no built-in padding is applied).
            padding: Must be 0; manual padding is used.
            dilation: Dilation factor.
            groups: Group count.
            bias: Whether to learn a bias term.
            look_ahead: Right-context samples available at inference.
            device: Optional device placement.
            dtype: Optional parameter dtype.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # we handle padding explicitly
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if padding != 0:
            raise ValueError("Pass padding=0; manual padding is used.")
        if look_ahead < 0:
            raise ValueError("look_ahead must be >= 0")

        k = (
            self.kernel_size[0]
            if isinstance(self.kernel_size, tuple)
            else int(self.kernel_size)
        )
        if k < stride:
            raise ValueError("kernel_size must be >= stride")

        d = self.dilation[0] if isinstance(self.dilation, tuple) else int(self.dilation)
        R = (k - 1) * d  # total left-span if strictly causal
        A = int(look_ahead)
        if A > R:
            raise ValueError(f"look_ahead ({A}) cannot exceed (k-1)*d = {R}")
        self._R_total = R  # for introspection
        self.look_ahead = A
        self._Rp = R - A  # effective past needed when using look-ahead

    # -----------------------------
    # Training / full-sequence mode
    # -----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence pass with manual padding.

        Args:
            x: Input tensor of shape (B, C_in, T).

        Returns:
            Tensor of shape (B, C_out, T_out) matching nn.Conv1d output.
        """
        if self._Rp or self.look_ahead:
            x = F.pad(x, (self._Rp, self.look_ahead))
        return super().forward(x)  # parent has padding=0

    # -----------------------------
    # External-state streaming API
    # -----------------------------
    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Initialize streaming state.

        Args:
            batch_size: Batch size.
            device: Optional device override.
            dtype: Optional dtype override.

        Returns:
            Dict with tail/past context, stride phase, and empty look-ahead buffer.
        """
        device = device or self.weight.device
        dtype = dtype or self.weight.dtype
        if self._Rp > 0:
            tail = torch.zeros(
                batch_size, self.in_channels, self._Rp, device=device, dtype=dtype
            )
        else:
            tail = torch.zeros(
                batch_size, self.in_channels, 0, device=device, dtype=dtype
            )
        phase = torch.zeros((), device=device, dtype=torch.int64)
        buffer = torch.zeros(
            batch_size, self.in_channels, 0, device=device, dtype=dtype
        )
        return {"tail": tail, "phase": phase, "buffer": buffer}

    # @torch.no_grad()
    # def stream(
    #     self, x_chunk: torch.Tensor, state: Dict[str, Any]
    # ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    #     """
    #     Streaming step with look-ahead A.

    #     Args:
    #       x_chunk: (B, C_in, T) new inputs.
    #       state:   {'tail': (B, C_in, Rp), 'phase': int64 0-D tensor}

    #     Returns:
    #       y_chunk: (B, C_out, T_out) outputs from this chunk whose future is available
    #       new_state: updated {'tail','phase'}
    #     """
    #     assert "tail" in state and "phase" in state, "State must have {'tail','phase'}."
    #     tail: torch.Tensor = state["tail"]
    #     phase: int = int(state["phase"].item())

    #     B, Cin, T = x_chunk.shape
    #     # Provide left context explicitly (Rp), no built-in padding.
    #     x_cat = (
    #         torch.cat([tail, x_chunk], dim=-1) if self._Rp > 0 else x_chunk
    #     )  # (B, Cin, Rp+T)

    #     # Right-pad by A zeros to match training-time look-ahead.
    #     x_conv = (
    #         F.pad(x_cat, (0, self._A)) if self._A > 0 else x_cat
    #     )  # (B, Cin, Rp+T+A)

    #     # Unstrided conv; output length = (Rp+T+A) - (k-1)*d = (Rp+T+A) - (Rp + A) = T
    #     y_full = F.conv1d(
    #         x_conv,
    #         self.weight,
    #         self.bias,
    #         stride=1,
    #         padding=0,
    #         dilation=self.dilation,
    #         groups=self.groups,
    #     )  # (B, C_out, T)

    #     # Drop the last A conv steps: they depended on zero-padded future.
    #     y_valid = (
    #         y_full[..., : (y_full.size(-1) - self._A)] if self._A > 0 else y_full
    #     )  # (B, C_out, T-A)

    #     # Downsample with stride, aligned by carried phase.
    #     s = self.stride[0] if isinstance(self.stride, tuple) else int(self.stride)
    #     if s == 1:
    #         y_chunk = y_valid
    #         new_phase = 0
    #     else:
    #         y_chunk = y_valid[..., phase::s]
    #         # Advance phase by the number of valid unstrided steps we just consumed (T-A).
    #         new_phase = (phase + y_valid.size(-1)) % s

    #     # Update tail with the last Rp real inputs (from x_cat, not x_conv).
    #     new_tail = x_cat[..., -self._Rp :] if self._Rp > 0 else tail

    #     new_state = {
    #         "tail": new_tail,
    #         "phase": torch.tensor(new_phase, device=x_chunk.device, dtype=torch.int64),
    #     }
    #     return y_chunk, new_state

    # @torch.no_grad()
    # def init_state_alternate(
    #     self,
    #     batch_size: int,
    #     device: Optional[torch.device] = None,
    #     dtype: Optional[torch.dtype] = None,
    # ) -> Dict[str, Any]:
    #     """
    #     Same as init_state() but also prepares a buffer for stream_alternate().
    #     """
    #     base_state = self.init_state(batch_size, device=device, dtype=dtype)
    #     device = device or self.weight.device
    #     dtype = dtype or self.weight.dtype
    #     base_state["buffer"] = torch.zeros(
    #         batch_size, self.in_channels, 0, device=device, dtype=dtype
    #     )
    #     return base_state

    @torch.no_grad()
    def stream(
        self,
        x_chunk: torch.Tensor,
        state: Dict[str, Any],
        flush: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Streaming step that buffers look-ahead so chunked output matches forward().

        Args:
            x_chunk: Tensor (B, C_in, Tchunk) with new samples.
            state: Dict with {'tail','phase','buffer'} from init_state/previous call.
            flush: If True, pad the buffer with look-ahead zeros and emit everything.

        Returns:
            (y_chunk, new_state) where y_chunk has only outputs whose future is known.
        """

        required = {"tail", "phase", "buffer"}
        if not required <= state.keys():
            raise ValueError(f"State must contain {required}.")

        # Unpack state: left context (tail), stride offset (phase), and held inputs.
        tail: torch.Tensor = state["tail"]
        phase: int = int(state["phase"].item())
        buffer: torch.Tensor = state["buffer"]

        # Append new chunk to buffered inputs (buffer holds the look-ahead region).
        B, Cin, _ = x_chunk.shape
        buffer = (
            torch.cat([buffer, x_chunk], dim=-1) if buffer.numel() else x_chunk.clone()
        )

        buffer_len = buffer.size(-1)
        # Decide how many raw inputs we can safely emit.
        # - Normal: emit everything except the last A inputs.
        # - Flush: emit all buffered inputs and account for padding of A zeros.
        emit_inputs = (
            buffer_len + self.look_ahead
            if flush
            else max(buffer_len - self.look_ahead, 0)
        )

        if emit_inputs > 0:
            buffer_for_conv = buffer
            if flush and self.look_ahead > 0:
                # On the final call, pad right by A zeros to mimic forward() padding.
                pad = torch.zeros(
                    B, Cin, self.look_ahead, device=x_chunk.device, dtype=x_chunk.dtype
                )
                buffer_for_conv = torch.cat([buffer, pad], dim=-1)

            # Provide past context via tail and run unstrided convolution.
            x_conv = torch.cat([tail, buffer_for_conv], dim=-1)
            y_full = F.conv1d(
                x_conv,
                self.weight,
                self.bias,
                stride=1,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
            # Keep only the positions whose future context is available.
            y_valid = y_full[..., :emit_inputs]
        else:
            y_valid = torch.zeros(
                B, self.out_channels, 0, device=x_chunk.device, dtype=self.weight.dtype
            )

        s = self.stride[0] if isinstance(self.stride, tuple) else int(self.stride)
        if s == 1:
            y_chunk = y_valid
            new_phase = 0
        else:
            # Align to the stride grid: skip until the next stride-aligned index.
            offset = (s - phase) % s
            if y_valid.size(-1) <= offset:
                y_chunk = y_valid[..., :0]
            else:
                y_chunk = y_valid[..., offset::s]
            new_phase = (phase + y_valid.size(-1)) % s

        if emit_inputs > 0:
            # Drop emitted inputs from buffer and update tail with latest real samples.
            consumed = buffer[..., :emit_inputs]
            buffer = buffer[..., emit_inputs:]
            new_tail = torch.cat([tail, consumed], dim=-1)
        else:
            new_tail = tail

        if new_tail.size(-1) > self._Rp:
            new_tail = new_tail[..., -self._Rp :]

        if flush:
            # Nothing left pending after final flush.
            buffer = buffer[..., 0:0]

        new_state = {
            "tail": new_tail,
            "phase": torch.tensor(new_phase, device=x_chunk.device, dtype=torch.int64),
            "buffer": buffer,
        }
        return y_chunk, new_state

    @property
    def receptive_left_total(self) -> int:
        """Total left reach with strictly causal setup: (k-1)*d."""
        return self._R_total

    @property
    def receptive_left_effective(self) -> int:
        """Past actually needed when using look-ahead: Rp = (k-1)*d - A."""
        return self._Rp


class StreamingCausalConvTranspose1d(nn.ConvTranspose1d):
    """
    Streaming wrapper for ConvTranspose1d (causal deconvolution).

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
        stride: Stride (padding/output_padding fixed to 0).
        dilation: Dilation factor.
        groups: Group count.
        bias: Optional bias.
        weight: Layer weights.
        overlap_out: Trailing overlap length between chunks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            device=device,
            dtype=dtype,
        )
        if padding != 0 or output_padding != 0:
            raise ValueError("Use padding=0 and output_padding=0 for causal streaming.")

        # Normalize ints
        self._k = (
            self.kernel_size[0]
            if isinstance(self.kernel_size, tuple)
            else int(self.kernel_size)
        )
        self._d = (
            self.dilation[0] if isinstance(self.dilation, tuple) else int(self.dilation)
        )
        self._s = self.stride[0] if isinstance(self.stride, tuple) else int(self.stride)

        # Overlap length in output samples between consecutive blocks
        self._overlap = max(0, self._d * (self._k - 1) + 1 - self._s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence pass (padding/output_padding fixed to 0).

        Args:
            x: Input tensor (B, C_in, T).

        Returns:
            Tensor (B, C_out, T_out) from nn.ConvTranspose1d.
        """
        return super().forward(x)

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Initialize overlap buffer for streaming ConvTranspose1d.

        Args:
            batch_size: Batch size.
            device: Optional device override.
            dtype: Optional dtype override.

        Returns:
            Dict {'overlap': tensor of shape (B, C_out, overlap_len)}.
        """
        device = device or self.weight.device
        dtype = dtype or self.weight.dtype
        if self._overlap > 0:
            overlap = torch.zeros(
                batch_size, self.out_channels, self._overlap, device=device, dtype=dtype
            )
        else:
            overlap = torch.zeros(
                batch_size, self.out_channels, 0, device=device, dtype=dtype
            )
        return {"overlap": overlap}

    @torch.no_grad()
    def stream(
        self, x_chunk: torch.Tensor, state: Dict[str, Any], flush: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process one chunk in streaming mode using overlap–add.

        Args:
            x_chunk: Tensor (B, C_in, Tchunk).
            state: Dict {'overlap': trailing overlap from previous chunk}.
            flush: If True, emit the entire block and clear overlap (end of stream).

        Returns:
            (y_emit, new_state) where y_emit is ready-to-emit samples and new_state carries overlap.
        """
        assert "overlap" in state, "State must contain 'overlap'."
        overlap_prev: torch.Tensor = state["overlap"]

        B, Cin, T = x_chunk.shape
        # 1) Raw block deconv
        y_block = F.conv_transpose1d(
            x_chunk,
            self.weight,
            self.bias,
            stride=self._s,
            padding=0,
            output_padding=0,
            dilation=self._d,
            groups=self.groups,
        )
        # y_block length = (T - 1)*s + d*(k-1) + 1

        # 2) Add previous overlap into front
        O = overlap_prev.size(-1)
        if O > 0:
            if y_block.size(-1) < O:
                y_block = F.pad(y_block, (0, O - y_block.size(-1)))
            y_block[..., :O] += overlap_prev

        # 3) Emit hop = T * stride samples
        hop = T * self._s
        if flush:
            y_emit = y_block
            tail = y_block[..., :0]  # empty overlap
        else:
            y_emit = y_block[..., :hop]

            # 4) Keep tail = y_block[..., hop:] as new overlap
            tail = y_block[..., hop:]
            if self.bias is not None:
                tail = tail - self.bias.view(1, -1, 1)
            if tail.size(-1) < self._overlap:
                tail = F.pad(tail, (0, self._overlap - tail.size(-1)))
            elif tail.size(-1) > self._overlap:
                tail = tail[..., : self._overlap]

        new_state = {"overlap": tail}
        return y_emit, new_state

    @torch.no_grad()
    def init_state_alternate(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Initialize state for the exact streaming path (recomputes full history).

        Args:
            batch_size: Batch size.
            device: Optional device override.
            dtype: Optional dtype override.

        Returns:
            Dict {'buffer': concatenated past inputs, 'emitted': outputs already returned}.
        """
        device = device or self.weight.device
        dtype = dtype or self.weight.dtype
        buffer = torch.zeros(
            batch_size, self.in_channels, 0, device=device, dtype=dtype
        )
        emitted = torch.zeros((), device=device, dtype=torch.int64)
        return {"buffer": buffer, "emitted": emitted}

    @torch.no_grad()
    def stream_alternate(
        self,
        x_chunk: torch.Tensor,
        state: Dict[str, Any],
        flush: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Numerically exact streaming by replaying ConvTranspose1d on all buffered input.
        More expensive than stream() but matches forward() exactly.

        Args:
            x_chunk: Tensor (B, C_in, Tchunk) of new inputs.
            state: Dict {'buffer': accumulated inputs, 'emitted': int64 count already output}.
            flush: If True, emit all remaining samples and clear the buffer.

        Returns:
            (y_emit, new_state) where y_emit is the newly available slice.
        """

        required = {"buffer", "emitted"}
        if not required <= state.keys():
            raise ValueError(f"State must contain {required}.")

        buffer = state["buffer"]
        emitted = int(state["emitted"].item())

        if buffer.numel():
            buffer = torch.cat([buffer, x_chunk], dim=-1)
        else:
            buffer = x_chunk.clone()

        y_full = F.conv_transpose1d(
            buffer,
            self.weight,
            self.bias,
            stride=self._s,
            padding=0,
            output_padding=0,
            dilation=self._d,
            groups=self.groups,
        )

        total_inputs = buffer.size(-1)
        safe_len = (
            y_full.size(-1) if flush else min(y_full.size(-1), total_inputs * self._s)
        )
        safe_len = max(safe_len, emitted)

        if safe_len > emitted:
            y_emit = y_full[..., emitted:safe_len]
        else:
            y_emit = y_full[..., :0]

        new_emitted = torch.tensor(safe_len, device=x_chunk.device, dtype=torch.int64)
        if flush:
            # Reset buffer so the object can be reused if needed.
            buffer = buffer[..., 0:0]
            new_emitted = torch.zeros((), device=x_chunk.device, dtype=torch.int64)

        new_state = {"buffer": buffer, "emitted": new_emitted}
        return y_emit, new_state

    @property
    def overlap_out(self) -> int:
        return self._overlap

    # ---------------------------


# Paired streaming demo
# ---------------------------


def paired_stream_demo(
    B=1,
    Cin=1,
    Cmid=8,
    Cout=1,
    T=32,
    chunk=8,
    k_an=5,
    s_an=2,
    d_an=1,
    look_ahead=1,  # analysis
    k_sy=4,
    s_sy=2,
    d_sy=1,  # synthesis (match stride)
    device="cpu",
    dtype=torch.float32,
):
    """
    Runs full-sequence (reference) vs streaming (chunked) through:
      analysis: Conv1d (causal, look_ahead=A)
      synthesis: ConvTranspose1d (causal deconv)
    Keeps constant end-to-end latency of A input samples.
    """
    torch.manual_seed(0)

    # Build layers
    analysis = StreamingCausalConv1d(
        Cin,
        Cmid,
        kernel_size=k_an,
        stride=s_an,
        dilation=d_an,
        bias=True,
        look_ahead=look_ahead,
    ).to(device=device, dtype=dtype)
    synthesis = StreamingCausalConvTranspose1d(
        Cmid, Cout, kernel_size=k_sy, stride=s_sy, dilation=d_sy, bias=True
    ).to(device=device, dtype=dtype)

    assert (
        s_an == s_sy
    ), "For constant overall rate and latency, set synthesis stride == analysis stride."

    # Random input
    x_full = torch.randn(B, Cin, T, device=device, dtype=dtype)

    # -------- Reference full pass --------
    # analysis forward (pads left Rp, right A)
    z_ref = analysis(x_full)  # (B, Cmid, ~T/s_an)
    y_ref = synthesis(z_ref)  # (B, Cout, ~T)
    # Trim to input-length + constant latency behavior:
    # With A look-ahead, the "valid" streaming output corresponds to input excluding the last A samples.
    # We'll compare streaming output (concatenated) to y_ref trimmed to the same emitted length.
    # The streaming pipeline below will tell us exactly how many were emitted.

    # -------- Streaming pass --------
    st_an = analysis.init_state(B, device=device, dtype=dtype)
    st_sy = synthesis.init_state(B, device=device, dtype=dtype)

    emitted = []
    t = 0
    A = look_ahead
    z_emits = []
    while t < T:
        x_chunk = x_full[..., t : t + chunk]  # (B, Cin, Tchunk)
        flush_an = True if (t + chunk) >= T else False
        # Analysis stream: emits at rate 1/s_an, *skipping* the last A unstrided positions of the chunk
        z_emit, st_an = analysis.stream(
            x_chunk, st_an, flush=flush_an
        )  # (B, Cmid, Zemit)
        z_emits.append(z_emit)

        # Synthesis stream: upsample back to input rate via hop = Zemit * s_sy
        flush = True if (t + chunk) >= T else False
        y_emit, st_sy = synthesis.stream(
            z_emit, st_sy, flush=flush
        )  # (B, Cout, Yemitted_now)

        emitted.append(y_emit)
        t += x_chunk.size(-1)
        print(
            f"t={t},\nx_chunk={x_chunk},\nz_emit={z_emit},\ny_emit={y_emit},\nx_chunk.size(-1)={x_chunk.size(-1)}, z_emit.size(-1)={z_emit.size(-1)}, y_emit.size(-1)={y_emit.size(-1)}"
        )

    y_stream = torch.cat(emitted, dim=-1)
    z_stream = torch.cat(z_emits, dim=-1) if z_emits else z_ref[..., :0]

    print("z_ref=", z_ref, z_ref.shape)
    print("z_stream=", z_stream, z_stream.shape)
    print("y_ref=", y_ref, y_ref.shape)
    print("y_stream=", y_stream, y_stream.shape)

    # -------- Make the reference comparable --------
    y_ref = y_ref[..., : y_stream.size(-1)]
    z_ref = z_ref[..., : z_stream.size(-1)]

    # Report error
    max_abs = (y_stream - y_ref).abs().max().item()
    max_abs_z = (z_stream - z_ref).abs().max().item()
    print(f"Shapes: y_stream={tuple(y_stream.shape)}, y_ref={tuple(y_ref.shape)}")
    print(f"Max abs diff: {max_abs:.3e}")
    print(f"Shapes: z_stream={tuple(z_stream.shape)}, z_ref={tuple(z_ref.shape)}")
    print(f"Max abs diff (analysis): {max_abs_z:.3e}")

    return y_stream, y_ref


def stream_conv_transpose_demo(
    B=1,
    Cin=1,
    Cout=1,
    T=14,
    k=4,
    s=2,
    d=1,
    chunk=7,
    device="cpu",
    dtype=torch.float32,
):
    torch.manual_seed(0)

    # Build layer (padding=0, output_padding=0 enforced by class)
    layer = StreamingCausalConvTranspose1d(
        in_channels=Cin,
        out_channels=Cout,
        kernel_size=k,
        stride=s,
        dilation=d,
        bias=True,
    ).to(device=device, dtype=dtype)

    # Random input
    x_full = torch.randn(B, Cin, T, device=device, dtype=dtype)

    # 1) Full (reference) output
    y_ref = layer(x_full)  # super().forward

    # 2) Streaming output (concat y_emit over chunks)
    state = layer.init_state(B, device=device, dtype=dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[..., t : t + chunk]
        flush = True if (t + chunk) >= T else False
        y_emit, state = layer.stream(x_chunk, state, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(-1)

    y_stream = torch.cat(outs, dim=-1)

    print("y_ref=", y_ref)
    print("y_stream=", y_stream)
    print("diff=", y_ref - y_stream)

    # Shapes must match exactly
    assert y_ref.shape == y_stream.shape, f"{y_ref.shape=} {y_stream.shape=}"
    # Numeric closeness
    atol = 1e-5 if dtype == torch.float32 else 5e-4
    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    max_abs = (y_ref - y_stream).abs().max().item()

    assert torch.allclose(y_ref, y_stream, atol=atol, rtol=rtol), f"max_abs={max_abs}"


def stream_conv_demo(
    B=1,
    Cin=1,
    Cout=1,
    T=21,
    k=5,
    s=2,
    d=1,
    look_ahead=1,
    chunk=7,
    device="cpu",
    dtype=torch.float32,
):
    torch.manual_seed(0)

    layer = StreamingCausalConv1d(
        in_channels=Cin,
        out_channels=Cout,
        kernel_size=k,
        stride=s,
        dilation=d,
        bias=True,
        look_ahead=look_ahead,
    ).to(device=device, dtype=dtype)

    x_full = torch.randn(B, Cin, T, device=device, dtype=dtype)

    # Reference full-sequence
    y_ref = layer(x_full)

    # Streaming with exact parity using stream_alternate + final flush
    state = layer.init_state(B, device=device, dtype=dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[..., t : t + chunk]
        flush = (t + x_chunk.size(-1)) >= T
        y_emit, state = layer.stream(x_chunk, state, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(-1)

    y_stream = torch.cat(outs, dim=-1)

    print("y_ref=", y_ref)
    print("y_stream=", y_stream)
    print("diff=", y_ref - y_stream)

    assert y_ref.shape == y_stream.shape, f"{y_ref.shape=} {y_stream.shape=}"
    atol = 1e-5 if dtype == torch.float32 else 5e-4
    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    max_abs = (y_ref - y_stream).abs().max().item()

    assert torch.allclose(y_ref, y_stream, atol=atol, rtol=rtol), f"max_abs={max_abs}"


if __name__ == "__main__":
    # Example run
    stream_conv_transpose_demo(device="cpu", dtype=torch.float32)
    stream_conv_demo(device="cpu", dtype=torch.float32)

    y_stream, y_ref = paired_stream_demo(
        B=2,
        Cin=1,
        Cmid=8,
        Cout=1,
        T=4096,
        chunk=320,
        k_an=5,
        s_an=4,
        d_an=1,
        look_ahead=1,
        k_sy=5,
        s_sy=4,
        d_sy=1,
        device="cpu",
        dtype=torch.float32,
    )
