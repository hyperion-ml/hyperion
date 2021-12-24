"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ..hyp_defs import float_cpu


class Splicing(object):
    """
    Class to do splicing for DNN input
    """

    def __init__(
        self,
        left_context=0,
        right_context=0,
        frame_shift=1,
        splice_pattern=None,
        pad_mode=None,
        **kwargs
    ):

        self.left_context = left_context
        self.right_context = right_context
        self.frame_shift = frame_shift
        self.splice_pattern = None
        if splice_pattern is not None:
            self.left_context = -splice_pattern[0]
            self.right_context = splice_pattern[-1]
            self.splice_pattern = splice_pattern + self.left_context
        self.pad_mode = pad_mode
        self.pad_width = ((self.left_context, self.right_context), (0, 0))
        self.pad_kwargs = kwargs

    def splice(self, x):
        if self.pad_mode is not None:
            x = np.pad(x, self.pad_width, **self.pad_kwargs)

        num_in_frames = x.shape[0]
        in_dim = x.shape[1]
        frame_span = self.left_context + self.right_context + 1
        num_out_frames = int(np.floor((num_in_frames - frame_span) / frame_shift + 1))

        if self.splice_pattern is None:
            out_dim = frame_span * in_dim
        else:
            out_dim = len(self.splice_pattern) * in_dim

        X = np.zeros((num_out_frames, out_dim), dtype=float_cpu())

        start = 0
        for i in range(num_out_frames):
            if self.splice_pattern is None:
                X[i, :] = x[start : start + frame_shift, :].ravel()
            else:
                splice_pattern = self.splice_pattern + self.start
                X[i, :] = x[splice_pattern, :].ravel()
            start += frame_shift

        return X
