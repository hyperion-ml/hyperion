"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import numpy as np
from jsonargparse import ActionParser, ArgumentParser


class FrameSelector:
    """Class to select speech frames.

    Attributes:
       tol_num_frames: Maximum tolerated mismatch between number of feature
         frames and number of VAD selector frames.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.feats.frame_selector import FrameSelector
      >>> x = np.random.randn(5, 3).astype(np.float32)
      >>> sel = np.array([True, False, True, True, False])
      >>> selector = FrameSelector(tol_num_frames=1)
      >>> x_sel = selector.select(x, sel)
    """

    def __init__(self, tol_num_frames: int = 3) -> None:
        """Initializes the frame selector.

        Args:
          tol_num_frames: Maximum tolerated absolute difference between feature
            frame count and VAD selector length.
        """
        if tol_num_frames < 0:
            raise ValueError(
                f"tol_num_frames must be >= 0, got {tol_num_frames!r}"
            )
        self.tol_num_frames = tol_num_frames

    def select(self, x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """Select speech frames.

        Args:
          x: Feature matrix with shape ``(num_frames, feat_dim)``.
          sel: Binary selector vector with shape ``(num_vad_frames,)``.

        Returns:
          Feature matrix with selected frames.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D feature matrix, got shape={x.shape}")
        sel = np.asarray(sel)
        if sel.ndim != 1:
            raise ValueError(f"sel must be a 1D selector vector, got shape={sel.shape}")
        sel = sel.astype(bool, copy=False)

        num_frames = x.shape[0]
        num_frames_vad = sel.shape[0]
        if num_frames == num_frames_vad:
            return x[sel, :]
        elif num_frames > num_frames_vad:
            if num_frames - num_frames_vad <= self.tol_num_frames:
                return x[:num_frames_vad, :][sel, :]
            else:
                raise ValueError(
                    "num_frames (%d) > num_frames_vad (%d) + tol (%d)"
                    % (num_frames, num_frames_vad, self.tol_num_frames)
                )
        else:
            if num_frames_vad - num_frames <= self.tol_num_frames:
                return x[sel[:num_frames], :]
            else:
                raise ValueError(
                    "num_frames_vad (%d) > num_frames (%d) + tol (%d)"
                    % (num_frames_vad, num_frames, self.tol_num_frames)
                )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters frame selector args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with frame-selector options.
        """
        valid_args = ("tol_num_frames",)

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return d

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """Adds frame-selector options to parser.

        Args:
          parser: Arguments parser.
          prefix: Options prefix.

        Returns:
          ``None``.
        """

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--tol-num-frames",
            type=int,
            default=3,
            help="maximum tolerated error between number of feature frames and VAD frames.",
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
