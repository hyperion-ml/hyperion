"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingCausalConv1d(nn.Conv1d):
    """
    Drop-in Conv1d with:
      • TRAINING (forward): left pad by Rp=(k-1)*d - A, right pad by A (look-ahead).
      • INFERENCE (stream): external state {'tail','phase'} with look-ahead A.
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
        device=None,
        dtype=None,
    ):
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
        d = self.dilation[0] if isinstance(self.dilation, tuple) else int(self.dilation)
        R = (k - 1) * d  # total left-span if strictly causal
        A = int(look_ahead)
        if A > R:
            raise ValueError(f"look_ahead ({A}) cannot exceed (k-1)*d = {R}")
        self._R_total = R  # for introspection
        self._A = A
        self._Rp = R - A  # effective past needed when using look-ahead

    # -----------------------------
    # Training / full-sequence mode
    # -----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence pass:
          pad_left = Rp = (k-1)*d - A
          pad_right = A
        """
        if self._Rp or self._A:
            x = F.pad(x, (self._Rp, self._A))
        return super().forward(x)  # parent has padding=0

    # -----------------------------
    # External-state streaming API
    # -----------------------------
    @torch.no_grad()
    def init_state(self, batch_size: int, device=None, dtype=None) -> Dict[str, Any]:
        """
        Returns:
          {
            'tail': (B, C_in, Rp), last Rp input samples (zeros if Rp=0),
            'phase': () int64 tensor in [0, stride-1] for strided alignment
          }
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
        return {"tail": tail, "phase": phase}

    @torch.no_grad()
    def stream(
        self, x_chunk: torch.Tensor, state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Streaming step with look-ahead A.

        Args:
          x_chunk: (B, C_in, T) new inputs.
          state:   {'tail': (B, C_in, Rp), 'phase': int64 0-D tensor}

        Returns:
          y_chunk: (B, C_out, T_out) outputs from this chunk whose future is available
          new_state: updated {'tail','phase'}
        """
        assert "tail" in state and "phase" in state, "State must have {'tail','phase'}."
        tail: torch.Tensor = state["tail"]
        phase: int = int(state["phase"].item())

        B, Cin, T = x_chunk.shape
        # Provide left context explicitly (Rp), no built-in padding.
        x_cat = (
            torch.cat([tail, x_chunk], dim=-1) if self._Rp > 0 else x_chunk
        )  # (B, Cin, Rp+T)

        # Right-pad by A zeros to match training-time look-ahead.
        x_conv = (
            F.pad(x_cat, (0, self._A)) if self._A > 0 else x_cat
        )  # (B, Cin, Rp+T+A)

        # Unstrided conv; output length = (Rp+T+A) - (k-1)*d = (Rp+T+A) - (Rp + A) = T
        y_full = F.conv1d(
            x_conv,
            self.weight,
            self.bias,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )  # (B, C_out, T)

        # Drop the last A conv steps: they depended on zero-padded future.
        y_valid = (
            y_full[..., : (y_full.size(-1) - self._A)] if self._A > 0 else y_full
        )  # (B, C_out, T-A)

        # Downsample with stride, aligned by carried phase.
        s = self.stride[0] if isinstance(self.stride, tuple) else int(self.stride)
        if s == 1:
            y_chunk = y_valid
            new_phase = 0
        else:
            y_chunk = y_valid[..., phase::s]
            # Advance phase by the number of valid unstrided steps we just consumed (T-A).
            new_phase = (phase + y_valid.size(-1)) % s

        # Update tail with the last Rp real inputs (from x_cat, not x_conv).
        new_tail = x_cat[..., -self._Rp :] if self._Rp > 0 else tail

        new_state = {
            "tail": new_tail,
            "phase": torch.tensor(new_phase, device=x_chunk.device, dtype=torch.int64),
        }
        return y_chunk, new_state

    @property
    def receptive_left_total(self) -> int:
        """Total left reach with strictly causal setup: (k-1)*d."""
        return self._R_total

    @property
    def look_ahead(self) -> int:
        return self._A

    @property
    def receptive_left_effective(self) -> int:
        """Past actually needed when using look-ahead: Rp = (k-1)*d - A."""
        return self._Rp

    import torch


from typing import Any, Dict, Tuple

import torch.nn as nn
import torch.nn.functional as F


class StreamingCausalConvTranspose1d(nn.ConvTranspose1d):
    """
    Streaming wrapper for ConvTranspose1d (causal deconvolution).

    TRAINING / FULL-SEQUENCE:
        Calls the parent nn.ConvTranspose1d.forward (with padding=0, output_padding=0).

    STREAMING:
        Maintains an 'overlap' buffer to stitch consecutive blocks via overlap–add.
        State format:
          state = {'overlap': (B, C_out, O)}
        where O = number of overlapping output samples between blocks.

    Notes:
        - Use padding=0 and output_padding=0 for clean causal streaming.
        - No stride phase needed; hop size is deterministic (T * stride).
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
        device=None,
        dtype=None,
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

    # -----------------------------
    # Training / full-sequence mode
    # -----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    # -----------------------------
    # External-state streaming API
    # -----------------------------
    @torch.no_grad()
    def init_state(self, batch_size: int, device=None, dtype=None) -> Dict[str, Any]:
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
        self, x_chunk: torch.Tensor, state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process one chunk in streaming mode.

        Args:
            x_chunk: (B, C_in, T)
            state:   {'overlap': (B, C_out, O)}

        Returns:
            y_emit: (B, C_out, T_out)   outputs to emit now
            new_state: updated {'overlap': ...}
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
        y_emit = y_block[..., :hop]

        # 4) Keep tail = y_block[..., hop:] as new overlap
        tail = y_block[..., hop:]
        if tail.size(-1) < self._overlap:
            tail = F.pad(tail, (0, self._overlap - tail.size(-1)))
        elif tail.size(-1) > self._overlap:
            tail = tail[..., : self._overlap]

        new_state = {"overlap": tail}
        return y_emit, new_state

    # -------------
    # Introspection
    # -------------
    @property
    def overlap_out(self) -> int:
        return self._overlap

    # ---------------------------


# Paired streaming demo
# ---------------------------


def paired_stream_demo(
    B=2,
    Cin=1,
    Cmid=8,
    Cout=1,
    T=4096,
    chunk=320,
    k_an=5,
    s_an=4,
    d_an=1,
    look_ahead=8,  # analysis
    k_sy=5,
    s_sy=4,
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
    while t < T:
        x_chunk = x_full[..., t : t + chunk]  # (B, Cin, Tchunk)
        # Analysis stream: emits at rate 1/s_an, *skipping* the last A unstrided positions of the chunk
        z_emit, st_an = analysis.stream(x_chunk, st_an)  # (B, Cmid, Zemit)

        # Synthesis stream: upsample back to input rate via hop = Zemit * s_sy
        y_emit, st_sy = synthesis.stream(z_emit, st_sy)  # (B, Cout, Yemitted_now)

        # Implement constant end-to-end latency of A input samples:
        # Delay the very first emission by exactly A samples (one-time buffer).
        if A > 0 and len(emitted) == 0:
            # prepend zeros to enforce initial latency (alternative: carry a small start-up queue)
            pad = torch.zeros(B, Cout, A, device=device, dtype=dtype)
            emitted.append(pad)

        emitted.append(y_emit)
        t += x_chunk.size(-1)

    # Flush the final synthesis overlap tail
    tail_sy = st_sy["overlap"]
    if tail_sy.numel() > 0 and tail_sy.size(-1) > 0:
        emitted.append(tail_sy)

    y_stream = torch.cat(emitted, dim=-1)

    # -------- Make the reference comparable --------
    # Streaming emits: (initial A zeros) + [perfect reconstruction of y_ref without its last A “future-dependent” samples]
    # So align by trimming y_ref's last A samples and then left-padding by A zeros:
    if A > 0:
        y_ref_effective = torch.cat(
            [
                torch.zeros(B, Cout, A, device=device, dtype=dtype),
                y_ref[..., : y_ref.size(-1) - A],
            ],
            dim=-1,
        )
    else:
        y_ref_effective = y_ref

    # Match lengths
    minL = min(y_stream.size(-1), y_ref_effective.size(-1))
    y_stream = y_stream[..., :minL]
    y_ref_effective = y_ref_effective[..., :minL]

    # Report error
    max_abs = (y_stream - y_ref_effective).abs().max().item()
    print(
        f"Shapes: y_stream={tuple(y_stream.shape)}, y_ref_eff={tuple(y_ref_effective.shape)}"
    )
    print(f"Max abs diff: {max_abs:.3e}")

    return y_stream, y_ref_effective


def _run_once(
    B=2,
    Cin=3,
    Cout=4,
    T=1024,
    k=5,
    s=2,
    d=1,
    chunk=123,
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
        y_emit, state = layer.stream(x_chunk, state)
        outs.append(y_emit)
        t += x_chunk.size(-1)
    # flush any remaining overlap (should be exactly overlap length)
    overlap_tail = state["overlap"]
    if overlap_tail.numel() > 0 and overlap_tail.size(-1) > 0:
        outs.append(overlap_tail)

    y_stream = torch.cat(outs, dim=-1)

    # Shapes must match exactly
    assert y_ref.shape == y_stream.shape, f"{y_ref.shape=} {y_stream.shape=}"

    # Numeric closeness
    atol = 1e-5 if dtype == torch.float32 else 5e-4
    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    max_abs = (y_ref - y_stream).abs().max().item()
    assert torch.allclose(y_ref, y_stream, atol=atol, rtol=rtol), f"max_abs={max_abs}"


if __name__ == "__main__":
    # Example run
    _run_once(device="cpu", dtype=torch.float32)

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
        look_ahead=8,
        k_sy=5,
        s_sy=4,
        d_sy=1,
        device="cpu",
        dtype=torch.float32,
    )
