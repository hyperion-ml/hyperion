"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils import ClassInfo, SegmentSet
from ...utils.misc import filter_func_args
from .hyper_sampler import HyperSampler


class ClassWeightedRandomSegChunkSampler(HyperSampler):
    """
    Class-balanced chunk sampler that randomly samples fixed-length chunks
    from segments grouped by class labels.

    Supports uniform or data-prior-based sampling of classes and segments.
    Allows control over the number of classes, segments, and chunks per batch.
    Supports optional hard negative mining using a class affinity matrix.

    Attributes:
        segments (SegmentSet): Table of segments to sample from.
        class_info (ClassInfo): Class table with one entry per class.
        min_chunk_length (float): Minimum chunk duration in seconds.
        max_chunk_length (float): Maximum chunk duration in seconds.
        min_batch_size (int): Minimum number of samples in a batch.
        max_batch_size (int): Maximum number of samples in a batch.
        avg_batch_size (float): Average per-GPU batch size used to estimate epoch length.
        var_batch_size (bool): Whether chunk length changes the batch size.
        num_chunks_per_seg_epoch (Union[str, int, float]): Number of chunks per segment in an epoch or "auto".
        num_segs_per_class (int): Number of segments to sample per class in a batch.
        num_chunks_per_seg (int): Number of chunks per segment in a batch.
        weight_exponent (float): Exponent applied to class weights.
        weight_mode (str): One of ["custom", "uniform", "data-prior"].
        seg_weight_mode (str): One of ["uniform", "data-prior"].
        num_hard_prototypes (int): Number of hard prototype classes to sample (used with affinity matrix).
        hard_prototypes (Optional[torch.Tensor]): Top-K hard prototype indices for each class.
        class_name (str): Column name for class ID in segments.
        length_name (str): Column name for duration in segments.
        batch (int): Number of batches already yielded in the current epoch.
        counts (Dict[str, int]): Debug counter for sampled class IDs.
        map_class_to_segs_idx (Dict[str, np.ndarray]): Cached segment row indices by
            class ID.
        _seg_lengths_np (np.ndarray): Cached segment durations.
        _seg_ids_np (np.ndarray): Cached segment IDs.
    """

    def __init__(
        self,
        segments: SegmentSet,
        class_info: ClassInfo,
        min_chunk_length: float,
        max_chunk_length: Optional[float] = None,
        *,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_batch_length: Optional[float] = None,
        num_chunks_per_seg_epoch: Union[str, int, float] = "auto",
        num_segs_per_class: int = 1,
        num_chunks_per_seg: int = 1,
        weight_exponent: float = 1.0,
        weight_mode: str = "custom",
        seg_weight_mode: str = "uniform",
        num_hard_prototypes: int = 0,
        affinity_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        class_name: str = "class_id",
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        iters_per_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        seed: int = 1234,
    ) -> None:
        """
        Initialize the sampler.

        Args:
            segments: Segment table to sample from.
            class_info: Class metadata table containing class IDs, indexes, and weights.
            min_chunk_length: Minimum sampled chunk duration in seconds.
            max_chunk_length: Maximum sampled chunk duration in seconds. If ``None``,
                uses ``min_chunk_length``.
            min_batch_size: Minimum number of chunks per GPU batch.
            max_batch_size: Optional maximum number of chunks per GPU batch.
            max_batch_length: Optional maximum total chunk duration per GPU batch.
            num_chunks_per_seg_epoch: Number of chunks per segment per epoch, or
                ``"auto"`` to infer it from average segment length.
            num_segs_per_class: Number of segments to sample for each selected class.
            num_chunks_per_seg: Number of chunks to sample from each selected segment.
            weight_exponent: Exponent applied to class weights after selecting
                ``weight_mode``.
            weight_mode: Class weighting strategy: ``"custom"``, ``"uniform"``, or
                ``"data-prior"``.
            seg_weight_mode: Segment weighting strategy within each class:
                ``"uniform"`` or ``"data-prior"``.
            num_hard_prototypes: Number of hard prototype classes to expand each
                sampled class into when hard prototype mining is enabled.
            affinity_matrix: Optional class affinity matrix used for hard prototype
                mining.
            class_name: Column in ``segments`` that contains class IDs.
            length_name: Column in ``segments`` that contains segment durations.
            max_batches_per_epoch: Optional maximum number of yielded batches per epoch.
            shuffle: Whether to vary the sampler seed by epoch.
            iters_per_epoch: Deprecated alias for ``num_chunks_per_seg_epoch``.
            batch_size: Deprecated alias for ``min_batch_size``.
            seed: Base random seed.
        """
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=shuffle, seed=seed
        )
        if len(segments) == 0:
            raise ValueError("segments must contain at least one row.")
        if len(class_info) == 0:
            raise ValueError("class_info must contain at least one row.")
        if min_chunk_length <= 0:
            raise ValueError(
                f"min_chunk_length must be positive, got {min_chunk_length}."
            )
        if max_chunk_length is not None and max_chunk_length < min_chunk_length:
            raise ValueError(
                "max_chunk_length must be greater than or equal to min_chunk_length."
            )
        if num_segs_per_class <= 0:
            raise ValueError(
                f"num_segs_per_class must be positive, got {num_segs_per_class}."
            )
        if num_chunks_per_seg <= 0:
            raise ValueError(
                f"num_chunks_per_seg must be positive, got {num_chunks_per_seg}."
            )
        if num_hard_prototypes < 0:
            raise ValueError(
                "num_hard_prototypes must be non-negative, "
                f"got {num_hard_prototypes}."
            )
        if max_batch_length is not None and max_batch_length <= 0:
            raise ValueError(
                f"max_batch_length must be positive, got {max_batch_length}."
            )
        self.class_name = class_name
        self.length_name = length_name
        self.segments = segments
        self.class_info = class_info
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )

        # computing min-batch-size
        if batch_size is not None:
            if min_batch_size != 1:
                logging.warning(
                    "Both batch_size and min_batch_size provided; using batch_size as min_batch_size."
                )
            min_batch_size = batch_size

        if min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be positive, got {min_batch_size}.")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")

        # computing max-batch-size
        if max_batch_length is None:
            min_batch_size = max(
                num_segs_per_class * num_chunks_per_seg, min_batch_size
            )
            max_batch_size_0 = int(
                min_batch_size * self.max_chunk_length / self.min_chunk_length
            )
        else:
            max_batch_size_0 = int(max_batch_length / self.min_chunk_length)
            min_batch_size = int(max_batch_length / self.max_chunk_length)
            min_required_batch_size = num_segs_per_class * num_chunks_per_seg
            if min_required_batch_size > min_batch_size:
                raise ValueError(
                    "Inconsistent segment/chunk configuration: "
                    f"num_segs_per_class={num_segs_per_class}, "
                    f"num_chunks_per_seg={num_chunks_per_seg}, "
                    f"min_batch_size={min_batch_size}."
                )

        max_batch_size = (
            max_batch_size_0
            if max_batch_size is None
            else min(max_batch_size_0, max_batch_size)
        )
        if min_batch_size > max_batch_size:
            raise ValueError(
                "min_batch_size must be less than or equal to max_batch_size, "
                f"got min_batch_size={min_batch_size}, max_batch_size={max_batch_size}."
            )

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.avg_batch_size = (min_batch_size + max_batch_size) / 2
        self.var_batch_size = self.min_batch_size != self.max_batch_size

        self.num_segs_per_class = num_segs_per_class
        self.num_chunks_per_seg = num_chunks_per_seg

        self.weight_exponent = weight_exponent
        self.weight_mode = weight_mode
        self.seg_weight_mode = seg_weight_mode
        if self.weight_mode not in ("custom", "uniform", "data-prior"):
            raise ValueError(
                "weight_mode must be one of 'custom', 'uniform', or 'data-prior', "
                f"got {self.weight_mode}."
            )
        if self.seg_weight_mode not in ("uniform", "data-prior"):
            raise ValueError(
                "seg_weight_mode must be one of 'uniform' or 'data-prior', "
                f"got {self.seg_weight_mode}."
            )

        self.num_hard_prototypes = num_hard_prototypes
        self.batch = 0

        # compute the number of batches / epoch
        # legacy config parameter
        num_chunks_per_seg_epoch = (
            iters_per_epoch if iters_per_epoch is not None else num_chunks_per_seg_epoch
        )
        self._set_num_chunks_per_seg_epoch(num_chunks_per_seg_epoch)
        self._compute_len()

        # # fast mapping from classes to segments
        # self.map_class_to_segs = self.segments.df[
        #     ["id", self.class_name, self.length_name]
        # ]
        # self.map_class_to_segs.set_index(self.class_name, drop=False, inplace=True)

        self._gather_class_info()
        self._set_class_weights()

        self.set_hard_prototypes(affinity_matrix)

        logging.info(
            (
                "sampler batches/epoch=%d min-batch-size=%d, max-batch-size=%d "
                "avg-batch-size/gpu=%.2f avg-classes/batch=%.2f  samples/(seg*epoch)=%d"
            ),
            self._len,
            self.min_batch_size,
            self.max_batch_size,
            self.avg_batch_size,
            self.avg_batch_size / num_segs_per_class / num_chunks_per_seg,
            self.num_chunks_per_seg_epoch,
        )

        self.counts = {}

    def _set_seed(self) -> None:
        """Initialize the random number generator for the current epoch."""
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.rank)
        else:
            self.rng.manual_seed(self.seed + 100 * self.rank)

    def _set_num_chunks_per_seg_epoch(
        self, num_chunks_per_seg_epoch: Union[str, int, float]
    ) -> None:
        """
        Set the number of chunks per segment per epoch.

        Args:
            num_chunks_per_seg_epoch: Number of chunks per segment, or ``"auto"``
                to infer it from average segment and chunk durations.
        """
        if num_chunks_per_seg_epoch == "auto":
            self._compute_num_chunks_per_seg_epoch_auto()
        else:
            if not isinstance(num_chunks_per_seg_epoch, (int, float)):
                raise ValueError(
                    "num_chunks_per_seg_epoch must be a positive number or 'auto', "
                    f"got {num_chunks_per_seg_epoch!r}."
                )
            if num_chunks_per_seg_epoch <= 0:
                raise ValueError(
                    "num_chunks_per_seg_epoch must be positive, "
                    f"got {num_chunks_per_seg_epoch}."
                )
            self.num_chunks_per_seg_epoch = num_chunks_per_seg_epoch

    def _compute_num_chunks_per_seg_epoch_auto(self) -> None:
        """Automatically determines number of chunks per segment using average segment and chunk length."""
        segments = self.segments
        avg_seg_length = np.mean(segments[self.length_name])
        avg_chunk_length = (self.max_chunk_length + self.min_chunk_length) / 2
        self.num_chunks_per_seg_epoch = math.ceil(avg_seg_length / avg_chunk_length)
        logging.info(
            "num chunks per segment and epoch: %d", self.num_chunks_per_seg_epoch
        )

    def _compute_len(self) -> None:
        """Computes the total number of batches per epoch."""
        self._len = int(
            math.ceil(
                self.num_chunks_per_seg_epoch
                * len(self.segments)
                / self.avg_batch_size
                / self.world_size
            )
        )
        if self.max_batches_per_epoch is not None:
            self._len = min(self._len, self.max_batches_per_epoch)

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        Returns:
            Number of batches yielded by each distributed rank.
        """
        return self._len

    def _gather_class_info(self) -> None:
        """Gather per-class duration statistics and cached segment mappings."""
        # we need the maximum/minimum segment duration for each class.
        logging.info("Gathering class info for %d classes", len(self.class_info))
        # ---------- fast column lookup ----------
        self._dur_col = self.segments.get_col_idx(self.length_name)
        self._id_col = self.segments.get_col_idx("id")
        self._class_id_col = self.class_info.get_col_idx("id")
        seg_df = self.segments.df
        seg_class = seg_df[self.class_name].values
        seg_len = seg_df[self.length_name].values.astype(np.float32)
        seg_ids = seg_df["id"].values

        # group indices per class id
        by_class: defaultdict[str, list[int]] = defaultdict(list)
        for idx, cid in enumerate(seg_class):
            by_class[cid].append(idx)
        # convert to numpy once
        self.map_class_to_segs_idx = {
            cid: np.asarray(idxs, dtype=np.int32) for cid, idxs in by_class.items()
        }

        # compute class-level stats in one vectorised pass
        class_ids = self.class_info["id"].values
        n_classes = len(class_ids)
        max_dur = np.zeros(n_classes, dtype=np.float32)
        min_dur = np.zeros(n_classes, dtype=np.float32)
        tot_dur = np.zeros(n_classes, dtype=np.float32)

        for i, cid in enumerate(class_ids):
            idxs = self.map_class_to_segs_idx.get(cid)
            if idxs is not None and idxs.size:
                d = seg_len[idxs]
                max_dur[i] = d.max()
                min_dur[i] = d.min()
                tot_dur[i] = d.sum()
            else:  # empty class
                self.map_class_to_segs_idx[cid] = np.empty(0, dtype=np.int32)

        self.class_info["max_seg_duration"] = max_dur
        self.class_info["min_seg_duration"] = min_dur
        self.class_info["total_duration"] = tot_dur

        # map class-idx → id (tiny table, copy fast)
        self.map_class_idx_to_ids = self.class_info.df.set_index("class_idx")[["id"]]

        #   duration array for quick look-up during sampling
        self._seg_lengths_np = seg_len
        self._seg_ids_np = seg_ids

    def _set_class_weights(self) -> None:
        """Computes class sampling weights based on weight_mode and duration filters."""
        logging.info(
            "Setting class weights using mode=%s, exponent=%.2f",
            self.weight_mode,
            self.weight_exponent,
        )
        if self.weight_mode == "uniform":
            self.class_info.set_uniform_weights()
        elif self.weight_mode == "data-prior":
            weights = self.class_info["total_duration"].values
            self.class_info.set_weights(weights)

        if self.weight_exponent != 1.0:
            self.class_info.exp_weights(self.weight_exponent)

        zero_weight = self.class_info["max_seg_duration"] < self.min_chunk_length
        if np.any(zero_weight):
            logging.info(
                "%d classes skipped due to insufficient segment length.",
                zero_weight.sum(),
            )
            self.class_info.set_zero_weight(zero_weight)

        self.var_weights = np.any(
            self.segments[self.length_name] < self.max_chunk_length
        )

    @property
    def hard_prototype_mining(self) -> bool:
        """
        Whether hard prototype mining is active.

        Returns:
            ``True`` when more than one hard prototype is requested and an
            affinity matrix has initialized hard prototypes.
        """
        return self.num_hard_prototypes > 1 and self.hard_prototypes is not None

    def set_hard_prototypes(
        self, affinity_matrix: Union[torch.Tensor, np.ndarray, None] = None
    ) -> None:
        """
        Initialize hard prototype class indices from a class affinity matrix.

        Args:
            affinity_matrix: Class-by-class similarity matrix. If ``None``, hard
                prototype mining is disabled.
        """
        if affinity_matrix is None:
            self.hard_prototypes = None
            return

        logging.info("Setting hard prototypes using affinity matrix")
        affinity_matrix = torch.as_tensor(affinity_matrix).clone()
        if affinity_matrix.ndim != 2:
            raise ValueError(
                "affinity_matrix must be two-dimensional, "
                f"got shape={tuple(affinity_matrix.shape)}."
            )
        if affinity_matrix.size(0) != affinity_matrix.size(1):
            raise ValueError(
                "affinity_matrix must be square, "
                f"got shape={tuple(affinity_matrix.shape)}."
            )
        class_idx = self.class_info["class_idx"].values
        if np.any(class_idx < 0) or np.any(class_idx >= affinity_matrix.size(0)):
            raise ValueError(
                "class_info class_idx values must index affinity_matrix, "
                f"whose shape is {tuple(affinity_matrix.shape)}."
            )
        if self.num_hard_prototypes > affinity_matrix.size(1):
            raise ValueError(
                "num_hard_prototypes must be less than or equal to the number of "
                f"classes in affinity_matrix, got num_hard_prototypes={self.num_hard_prototypes}, "
                f"num_classes={affinity_matrix.size(1)}."
            )
        # don't sample hard negs from classes with zero weight or absent
        zero_w = self.class_info["weights"] == 0
        num_active_classes = int((~zero_w).sum())
        if self.num_hard_prototypes > num_active_classes:
            raise ValueError(
                "num_hard_prototypes must be less than or equal to the number of "
                "sampleable classes, "
                f"got num_hard_prototypes={self.num_hard_prototypes}, "
                f"num_sampleable_classes={num_active_classes}."
            )
        if np.any(zero_w):
            zero_w_idx = self.class_info.loc[zero_w, "class_idx"].values
            affinity_matrix[:, zero_w_idx] = -1000

        # keep only existing class indices
        valid_idx = self.class_info["class_idx"].values
        invalid_idx = np.setdiff1d(np.arange(affinity_matrix.size(1)), valid_idx)
        if invalid_idx.size:
            affinity_matrix[:, invalid_idx] = -1000
        # for i in range(affinity_matrix.size(1)):
        #     mask_i = self.class_info["class_idx"] == i
        #     if np.all(mask_i == 0):
        #         affinity_matrix[:, i] = -1000

        # hard prototypes for a class are itself and k-1 closest to it.
        self.hard_prototypes = torch.topk(
            affinity_matrix, self.num_hard_prototypes, dim=-1
        ).indices

    def get_hard_prototypes(
        self, class_idx: Union[int, np.ndarray], chunk_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Return top-K hard prototype indices for class indices.

        Args:
            class_idx: Class index or array of class indices.
            chunk_length: If provided, replace hard prototype classes that are too
                short for this chunk length with the corresponding input class.

        Returns:
            Flattened array of hard prototype class indices.
        """
        class_idx_np = np.asarray(class_idx)
        class_idx_t = torch.as_tensor(
            class_idx_np, dtype=torch.long, device=self.hard_prototypes.device
        )
        hard_class_idx = self.hard_prototypes[class_idx_t].cpu().numpy()

        if chunk_length is not None and self.var_weights:
            hard_class_ids = self.map_class_idx_to_ids.loc[
                hard_class_idx.flatten(), "id"
            ].values.reshape(hard_class_idx.shape)
            valid_hard_class = (
                self.class_info.loc[
                    hard_class_ids.flatten(), "max_seg_duration"
                ].values.reshape(hard_class_idx.shape)
                >= chunk_length
            )
            hard_class_idx = np.where(
                valid_hard_class, hard_class_idx, class_idx_np[..., np.newaxis]
            )

        return hard_class_idx.flatten()

    def _sample_chunk_length(self) -> float:
        """
        Sample a random chunk duration.

        Returns:
            Chunk duration in seconds.
        """
        if self.var_batch_size:
            return (
                torch.rand(size=(1,), generator=self.rng).item()
                * (self.max_chunk_length - self.min_chunk_length)
                + self.min_chunk_length
            )

        return self.min_chunk_length

    def _compute_batch_size(self, chunk_length: float) -> int:
        """
        Compute a batch size compatible with the sampled chunk length.

        Args:
            chunk_length: Sampled chunk duration in seconds.

        Returns:
            Number of chunks to sample in the batch.
        """
        batch_size = int(self.min_batch_size * self.max_chunk_length / chunk_length)
        return min(batch_size, self.max_batch_size)

    def _compute_num_classes_per_batch(self, batch_size: int) -> int:
        """
        Compute how many classes should be sampled for a batch size.

        Args:
            batch_size: Number of chunks in the batch.

        Returns:
            Number of class IDs to sample before optional hard prototype expansion.
        """
        num_classes = batch_size / self.num_segs_per_class / self.num_chunks_per_seg
        if self.hard_prototype_mining:
            num_classes /= self.num_hard_prototypes
        return int(math.ceil(num_classes))

    def _get_class_weights(self, chunk_length: float) -> torch.Tensor:
        """
        Return class sampling weights for the requested chunk length.

        Args:
            chunk_length: Sampled chunk duration in seconds.

        Returns:
            Tensor of class sampling probabilities.
        """
        class_weights = self.class_info["weights"].values.copy()
        # get classes where all segments are shorter than
        # chunk length and put weight to 0
        if self.var_weights:
            zero_mask = self.class_info["max_seg_duration"] < chunk_length
            if np.any(zero_mask):
                class_weights[zero_mask] = 0.0

        # renormalize weights
        weight_sum = class_weights.sum()
        if weight_sum <= 0:
            raise ValueError(
                f"No classes have segments long enough for chunk_length={chunk_length}."
            )
        class_weights /= weight_sum
        return torch.as_tensor(class_weights)

    def _sample_classes(self, num_classes: int, chunk_length: float) -> np.ndarray:
        """
        Randomly sample class IDs for a batch.

        Args:
            num_classes: Number of class IDs to sample before optional hard prototype
                expansion.
            chunk_length: Sampled chunk duration in seconds.

        Returns:
            Array of selected class IDs.
        """
        weights = self._get_class_weights(chunk_length)
        row_idx = torch.multinomial(
            weights,
            num_samples=num_classes,
            replacement=True,
            generator=self.rng,
        ).numpy()

        class_ids = self.class_info.iloc[row_idx, self._class_id_col].values
        if self.hard_prototype_mining:
            # map class ids to class indexes
            class_idx = self.class_info.loc[class_ids, "class_idx"].values
            class_idx = self.get_hard_prototypes(class_idx, chunk_length)
            # map back to class ids
            class_ids = self.map_class_idx_to_ids.loc[class_idx, "id"].values

        return class_ids

    def _sample_segs(self, class_ids: List[str], chunk_length: float) -> List[str]:
        """
        Sample segment IDs for each selected class.

        Args:
            class_ids: Class IDs selected for the batch.
            chunk_length: Sampled chunk duration in seconds.

        Returns:
            Segment IDs selected for chunk sampling.
        """

        # dur_col_idx, id_col_idx = self._dur_col, self._id_col
        seg_ids = []
        for c in class_ids:
            # for each class we sample segments longer than chunk length
            # get segments belonging to c
            # t1 = time.time()
            seg_idx_c = self.map_class_to_segs_idx[c]
            if not seg_idx_c.size:
                logging.error("no segments found with class=%s", c)
                continue
            # t2 = time.time()
            durs = self._seg_lengths_np[seg_idx_c]
            # filter segments that are too short
            # t2 = time.time()
            if self.class_info.loc[c, "min_seg_duration"] < chunk_length:
                mask = durs >= chunk_length
                seg_idx_c = seg_idx_c[mask]
                durs = durs[mask]

            # t3 = time.time()
            # sample num_segs_per_class random segments
            if len(seg_idx_c) == 0:
                logging.error("no segments found with class=%s dur=%d", c, chunk_length)
            if self.seg_weight_mode == "uniform":
                sel_idx = torch.randint(
                    low=0,
                    high=len(seg_idx_c),
                    size=(self.num_segs_per_class,),
                    generator=self.rng,
                ).numpy()

            elif self.seg_weight_mode == "data-prior":
                weights = durs / durs.sum()
                sel_idx = torch.multinomial(
                    torch.from_numpy(weights),
                    num_samples=self.num_segs_per_class,
                    replacement=True,
                    generator=self.rng,
                ).numpy()
                # t4 = time.time()
            else:
                raise ValueError(f"unknown seg-weight-mode={self.seg_weight_mode}")

            sel_seg_idx_c = seg_idx_c[sel_idx]
            sel_seg_ids_c = self._seg_ids_np[sel_seg_idx_c]
            # t5 = time.time()
            seg_ids.extend(sel_seg_ids_c)
            # t6 = time.time()
            # logging.info(
            #     "stime %f %f %f %f %f", t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5
            # )

        return seg_ids

    def _sample_chunks(
        self, seg_ids: Union[List[str], np.ndarray], chunk_length: float
    ) -> List[Tuple[str, float, float]]:
        """
        Sample chunk start positions from selected segments.

        Args:
            seg_ids: Segment IDs selected for the batch.
            chunk_length: Sampled chunk duration in seconds.

        Returns:
            Tuples of ``(segment_id, start_time, chunk_length)``.
        """
        lens = torch.as_tensor(
            self.segments.loc[seg_ids, self.length_name].values, dtype=torch.float32
        )
        scale = lens - chunk_length
        chunks = []
        for i in range(self.num_chunks_per_seg):
            starts = scale * torch.rand(len(lens), generator=self.rng)
            chunks.extend(
                [(sid, s.item(), chunk_length) for sid, s in zip(seg_ids, starts)]
            )
        return chunks

    def __next__(self) -> List[Tuple[str, float, float]]:
        """
        Sample and return a new batch of chunks.

        Returns:
            Tuples of ``(segment_id, start_time, chunk_length)``.
        """
        if self.batch >= self._len:
            raise StopIteration

        profile_next = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
        if profile_next:
            start_time = time.perf_counter()
        chunk_length = self._sample_chunk_length()
        if profile_next:
            chunk_length_time = time.perf_counter()
        batch_size = self._compute_batch_size(chunk_length)
        if profile_next:
            batch_size_time = time.perf_counter()
        num_classes = self._compute_num_classes_per_batch(batch_size)
        if profile_next:
            num_classes_time = time.perf_counter()
        class_ids = self._sample_classes(num_classes, chunk_length)
        if profile_next:
            class_ids_time = time.perf_counter()
        seg_ids = self._sample_segs(class_ids, chunk_length)
        if profile_next:
            seg_ids_time = time.perf_counter()
        chunks = self._sample_chunks(seg_ids, chunk_length)
        if profile_next:
            chunks_time = time.perf_counter()
            logging.debug(
                "%s __next__ batch=%d chunk-length=%.3f batch-size=%d "
                "num-classes=%d timings-ms: chunk-length=%.3f batch-size=%.3f "
                "num-classes=%.3f classes=%.3f segments=%.3f chunks=%.3f "
                "total=%.3f",
                __name__,
                self.batch,
                chunk_length,
                batch_size,
                num_classes,
                (chunk_length_time - start_time) * 1e3,
                (batch_size_time - chunk_length_time) * 1e3,
                (num_classes_time - batch_size_time) * 1e3,
                (class_ids_time - num_classes_time) * 1e3,
                (seg_ids_time - class_ids_time) * 1e3,
                (chunks_time - seg_ids_time) * 1e3,
                (chunks_time - start_time) * 1e3,
            )
        if self.batch == 0:
            logging.info("batch 0 uttidx=%s", str(chunks[:10]))

        self.batch += 1
        return chunks

    def __iter__(self) -> "ClassWeightedRandomSegChunkSampler":
        """
        Return this sampler as its own iterator.

        Returns:
            This sampler instance.
        """
        resume_batch = self.init_batch
        self.init_batch = 0
        super().__iter__()
        if resume_batch != 0:
            logging.info(
                "Replaying %d batches to resume ClassWeightedRandomSegChunkSampler.",
                resume_batch,
            )
            replay_start = time.monotonic()
            for _ in range(min(resume_batch, self._len)):
                try:
                    next(self)
                except StopIteration:
                    break
            logging.info(
                "Finished replaying %d batches to resume ClassWeightedRandomSegChunkSampler in %.2f seconds.",
                self.batch,
                time.monotonic() - replay_start,
            )
        return self

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter keyword arguments accepted by the sampler constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dictionary containing only constructor-compatible arguments.
        """
        return filter_func_args(ClassWeightedRandomSegChunkSampler.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """
        Adds command-line arguments for configuring the ClassWeightedRandomSegChunkSampler.

        These arguments define chunking behavior, class-balanced sampling, batching constraints,
        weight strategies, and various runtime controls.

        Args:
            parser (ArgumentParser): The argument parser instance to populate.
            prefix (Optional[str]): If provided, nests arguments under the given prefix key.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--min-chunk-length",
            type=float,
            default=4.0,
            help="Minimum duration (in seconds) of a sampled chunk.",
        )
        parser.add_argument(
            "--max-chunk-length",
            type=float,
            default=None,
            help="Maximum duration of a chunk. Defaults to --min-chunk-length if not set.",
        )

        parser.add_argument(
            "--min-batch-size",
            type=int,
            default=1,
            help="Minimum number of samples per batch per GPU.",
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=None,
            help="Maximum number of samples per batch. If not set, estimated from --max-batch-length.",
        )

        parser.add_argument(
            "--batch-size",
            default=None,
            type=int,
            help="(Deprecated) Use --min-batch-size instead.",
        )

        parser.add_argument(
            "--max-batch-length",
            "--max-batch-duration",
            dest="max_batch_length",
            type=float,
            default=None,
            help="Maximum total duration of all chunks in a batch. Overrides batch size if set.",
        )

        parser.add_argument(
            "--iters-per-epoch",
            default=None,
            type=lambda x: x if (x == "auto" or x is None) else float(x),
            help="(Deprecated) Use --num-chunks-per-seg-epoch instead.",
        )

        parser.add_argument(
            "--num-chunks-per-seg-epoch",
            default="auto",
            type=lambda x: x if x == "auto" else float(x),
            help="Number of chunks to draw per segment per epoch, or 'auto' to infer it.",
        )

        parser.add_argument(
            "--num-segs-per-class",
            type=int,
            default=1,
            help="Number of segments to sample per class in each batch.",
        )
        parser.add_argument(
            "--num-chunks-per-seg",
            type=int,
            default=1,
            help="Number of chunks to extract per segment in a batch.",
        )

        parser.add_argument(
            "--weight-exponent",
            type=float,
            default=1.0,
            help="Exponent to scale class weights: weight = weight ** exponent.",
        )
        parser.add_argument(
            "--weight-mode",
            default="custom",
            choices=["custom", "uniform", "data-prior"],
            help="Strategy to assign weights to each class: custom values, uniform weights, or based on total duration.",
        )

        parser.add_argument(
            "--seg-weight-mode",
            default="uniform",
            choices=["uniform", "data-prior"],
            help="Sampling strategy for segments within a class: uniform or duration-proportional.",
        )

        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help="Number of hard negative prototype classes to include per batch.",
        )

        parser.add_argument(
            "--max-batches-per-epoch",
            type=int,
            default=None,
            help="Optional maximum number of batches per epoch.",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="If True, enables epoch-level shuffling of segment order.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Random seed used for reproducible sampling.",
        )

        parser.add_argument(
            "--length-name",
            default="duration",
            help="Name of the column in the segment table that defines segment length.",
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="Name of the column in the segment table that defines the class label.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
