"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Any, Optional, Sequence

import numpy as np

from ..hyp_defs import float_cpu


class Splicing:
    """Frame splicing utility for DNN input features.

    Splicing concatenates neighboring frames around each time step.
    The context can be specified either with ``left_context/right_context``
    or with an explicit ``splice_pattern`` (relative frame offsets).
    """

    def __init__(
        self,
        left_context: int = 0,
        right_context: int = 0,
        frame_shift: int = 1,
        splice_pattern: Optional[Sequence[int]] = None,
        pad_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the splicing configuration.

        Args:
            left_context: Number of frames to take on the left.
            right_context: Number of frames to take on the right.
            frame_shift: Hop size (in frames) between consecutive outputs.
            splice_pattern: Optional list of relative frame offsets
                (e.g. ``[-2, 0, 2]``). If provided, it overrides left/right
                context.
            pad_mode: Padding mode for ``numpy.pad``. If ``None``, no padding
                is applied and only fully valid windows are emitted.
            **kwargs: Extra keyword arguments forwarded to ``numpy.pad``.
        """
        if frame_shift <= 0:
            raise ValueError(f"frame_shift must be > 0, got {frame_shift}")
        if left_context < 0 or right_context < 0:
            raise ValueError(
                "left_context and right_context must be >= 0, got "
                f"{left_context}, {right_context}"
            )

        self.left_context = left_context
        self.right_context = right_context
        self.frame_shift = frame_shift
        self.splice_pattern = None
        if splice_pattern is not None:
            p = np.asarray(splice_pattern, dtype=np.int64)
            if p.ndim != 1 or p.size == 0:
                raise ValueError(
                    f"splice_pattern must be a non-empty 1D sequence, got shape {p.shape}"
                )
            self.left_context = int(-np.min(p))
            self.right_context = int(np.max(p))
            # Convert relative offsets into non-negative local indices.
            self.splice_pattern = p + self.left_context
        self.pad_mode = pad_mode
        self.pad_width = ((self.left_context, self.right_context), (0, 0))
        self.pad_kwargs = kwargs

    def splice(self, x: np.ndarray) -> np.ndarray:
        """Apply frame splicing to a 2D feature matrix.

        Args:
            x: Input features of shape ``(num_frames, feat_dim)``.

        Returns:
            Spliced features of shape ``(num_out_frames, out_dim)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")

        if self.pad_mode is not None:
            x = np.pad(x, self.pad_width, mode=self.pad_mode, **self.pad_kwargs)

        num_in_frames = x.shape[0]
        in_dim = x.shape[1]
        frame_span = self.left_context + self.right_context + 1

        if self.splice_pattern is None:
            out_dim = frame_span * in_dim
        else:
            out_dim = len(self.splice_pattern) * in_dim

        if num_in_frames < frame_span:
            return np.zeros((0, out_dim), dtype=float_cpu())

        num_out_frames = 1 + (num_in_frames - frame_span) // self.frame_shift
        X = np.zeros((num_out_frames, out_dim), dtype=float_cpu())

        start = 0
        for i in range(num_out_frames):
            if self.splice_pattern is None:
                X[i, :] = x[start : start + frame_span, :].ravel()
            else:
                splice_pattern = self.splice_pattern + start
                X[i, :] = x[splice_pattern, :].ravel()
            start += self.frame_shift

        return X
