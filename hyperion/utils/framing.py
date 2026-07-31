"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Functions to create frames
"""

from typing import Any, Callable, Literal, Optional, Tuple, Union

import numpy as np

from ..hyp_defs import float_cpu


class Framing:
    """Class to create frames from signals or superframes from frame sequences.

    Attributes:
      frame_length: Length of the frames.
      frame_shift: Shift of the frames.
      pad_mode: padding mode, see numpy.pad, None means no padding
        One of the following string values or a user supplied function.

          'constant'
             Pads with a constant value.

          'edge'
             Pads with the edge values of array.

          'linear_ramp'
             Pads with the linear ramp between end_value and the array edge value.

          'maximum'
             Pads with the maximum value of all or part of the vector along each axis.

          'mean'
             Pads with the mean value of all or part of the vector along each axis.

          'median'
             Pads with the median value of all or part of the vector along each axis.

          'minimum'
             Pads with the minimum value of all or part of the vector along each axis.

          'reflect'
             Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.

          'symmetric'
             Pads with the reflection of the vector mirrored along the edge of the array.

          'wrap'
             Pads with the wrap of the vector along the axis. The first values
             are used to pad the end and the end values are used to pad the beginning.

          <function>
        Padding function, see Notes.

      pad_side: padding side {symmetric (default), left, right}.
      pad_width: Number of values padded to the edges of each axis.
          ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
          ((before, after),) yields same before and after pad for each axis.
          (pad,) or int is a shortcut for before = after = pad width for all axes.
      pad_kwargs: extra arguments for numpy.pad
    """

    def __init__(
        self,
        frame_length: int,
        frame_shift: int = 1,
        pad_mode: Optional[Union[str, Callable[..., np.ndarray]]] = None,
        pad_side: Literal["symmetric", "left", "right"] = "symmetric",
        **kwargs: Any,
    ) -> None:
        if frame_length <= 0:
            raise ValueError(f"frame_length must be > 0, got {frame_length}")
        if frame_shift <= 0:
            raise ValueError(f"frame_shift must be > 0, got {frame_shift}")

        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.pad_mode = pad_mode
        self.pad_width = None
        if self.pad_mode is not None:
            self.pad_width = self.create_pad_width(pad_side, frame_length, frame_shift)
        self.pad_kwargs = kwargs

    @staticmethod
    def create_pad_width(
        pad_side: Literal["symmetric", "left", "right"],
        frame_length: int,
        frame_shift: int,
    ) -> Tuple[int, int]:
        """Calculates the proper pad_with for left and rigth from the frame lengths and shift.

        Args:
          pad_side: symmetric, left, right.
          frame_length: Frame length.
          frame_shift: Frame shift.

        Returns:
          2D tuple with left and right pad width.
        """
        overlap = max(frame_length - frame_shift, 0)
        if pad_side == "symmetric":
            pad_width = (int(np.ceil(overlap / 2)), int(np.floor(overlap / 2)))
        elif pad_side == "left":
            pad_width = (int(overlap), 0)
        elif pad_side == "right":
            pad_width = (0, int(overlap))
        else:
            raise ValueError(f"Unknown pad_side={pad_side}")
        return pad_width

    def create_frames(self, x: np.ndarray) -> np.ndarray:
        """Create the frames.

        Args:
           x: 1D or 2D numpy array.

        Returns:
           2D numpy array.
             If x is 1D, each output frame (row) will contain frame_length samples from x.
             If x is 2D, each output frame (row) will contain frame_length rows from x.

        """
        if x.ndim not in (1, 2):
            raise ValueError(f"x must be a 1D or 2D array, got ndim={x.ndim}")

        if self.pad_mode is not None:
            x = self.apply_padding(x)

        if x.ndim == 1:
            num_samples = x.shape[0]
            in_dim = 1
        else:
            num_samples = x.shape[0]
            in_dim = x.shape[1]

        if num_samples < self.frame_length:
            num_out_frames = 0
        else:
            num_out_frames = 1 + (num_samples - self.frame_length) // self.frame_shift

        vec_x = x.ravel()
        out_dim = self.frame_length * in_dim
        X = np.zeros((num_out_frames, out_dim), dtype=float_cpu())

        start = 0
        stop = out_dim
        shift = in_dim * self.frame_shift
        for i in range(num_out_frames):
            X[i, :] = vec_x[start:stop]
            start += shift
            stop += shift

        return X

    def apply_padding(self, x: np.ndarray) -> np.ndarray:
        """Calls numpy.pad with the rigth arguments."""
        if self.pad_mode is None or self.pad_width is None:
            raise ValueError("padding requested but pad_mode is None")
        pad_width = self.pad_width
        if x.ndim == 2:
            pad_width = (pad_width, (0, 0))
        return np.pad(x, pad_width, mode=self.pad_mode, **self.pad_kwargs)
