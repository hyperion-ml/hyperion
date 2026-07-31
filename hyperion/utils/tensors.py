"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Sequence, Tuple

import numpy as np


def to3D_by_class(
    x: np.ndarray, class_ids: np.ndarray, max_length: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack frame-level samples into a 3D tensor grouped by class id.

    Args:
        x: 2D input array of shape ``(num_samples, feat_dim)``.
        class_ids: 1D integer class ids of shape ``(num_samples,)``.
        max_length: Maximum number of samples per class in the output. If 0,
            it is inferred from the largest class count.

    Returns:
        x3d: Array of shape ``(num_classes, max_length, feat_dim)``.
        sample_weight: Binary mask of shape ``(num_classes, max_length)``
            indicating valid entries in ``x3d``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    class_ids = np.asarray(class_ids)
    if class_ids.ndim != 1:
        raise ValueError(f"class_ids must be 1D, got shape {class_ids.shape}")
    if x.shape[0] != class_ids.shape[0]:
        raise ValueError(
            f"x.shape[0]={x.shape[0]} must equal class_ids.shape[0]={class_ids.shape[0]}"
        )
    if class_ids.size == 0:
        return (
            np.zeros((0, max_length, x.shape[1]), dtype=x.dtype),
            np.zeros((0, max_length), dtype=np.float32),
        )
    if not np.issubdtype(class_ids.dtype, np.integer):
        raise ValueError(f"class_ids must be integer dtype, got {class_ids.dtype}")
    if np.any(class_ids < 0):
        raise ValueError("class_ids must be >= 0")

    dim = x.shape[1]
    num_classes = int(np.max(class_ids)) + 1
    if max_length == 0:
        for i in range(num_classes):
            num_i = np.sum(class_ids == i)
            max_length = np.maximum(max_length, num_i)
    else:
        max_count = int(np.max(np.bincount(class_ids, minlength=num_classes)))
        if max_length < max_count:
            raise ValueError(
                f"max_length={max_length} is smaller than largest class count={max_count}"
            )

    x3d = np.zeros((num_classes, max_length, dim), dtype=x.dtype)
    sample_weight = np.zeros((num_classes, max_length), dtype=np.float32)
    for i in range(num_classes):
        idx = class_ids == i
        num_i = np.sum(idx)
        x3d[i, :num_i, :] = x[idx, :]
        sample_weight[i, :num_i] = 1.0

    return x3d, sample_weight


def to3D_by_seq(
    x: Sequence[np.ndarray], max_length: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack variable-length feature sequences into a padded 3D tensor.

    Args:
        x: Sequence of 2D arrays, each of shape ``(num_frames_i, feat_dim)``.
        max_length: Maximum sequence length in the output. If 0, inferred from
            the longest sequence.

    Returns:
        x3d: Array of shape ``(num_seqs, max_length, feat_dim)``.
        sample_weight: Binary mask of shape ``(num_seqs, max_length)``
            indicating valid frames in ``x3d``.
    """
    if len(x) == 0:
        raise ValueError("x must contain at least one sequence")
    for i, seq in enumerate(x):
        if seq.ndim != 2:
            raise ValueError(f"x[{i}] must be 2D, got shape {seq.shape}")

    dim = x[0].shape[1]
    num_seqs = len(x)
    for i, seq in enumerate(x):
        if seq.shape[1] != dim:
            raise ValueError(f"x[{i}] has feature dim {seq.shape[1]}, expected {dim}")

    if max_length == 0:
        for i in range(num_seqs):
            num_i = x[i].shape[0]
            max_length = np.maximum(max_length, num_i)
    else:
        max_seq_len = max(seq.shape[0] for seq in x)
        if max_length < max_seq_len:
            raise ValueError(
                f"max_length={max_length} is smaller than longest sequence={max_seq_len}"
            )

    x3d = np.zeros((num_seqs, max_length, dim), dtype=x[0].dtype)
    sample_weight = np.zeros((num_seqs, max_length), dtype=np.float32)
    for i in range(num_seqs):
        num_i = x[i].shape[0]
        x3d[i, :num_i, :] = x[i]
        sample_weight[i, :num_i] = 1.0

    return x3d, sample_weight
