"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from collections import OrderedDict as ODict
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist

MetricValue = Union[float, int, np.floating, np.integer, torch.Tensor]


class MetricAcc:
    """Accumulate running averages of scalar metrics across mini-batches.

    Supports distributed training by reducing batch metrics to rank 0 before
    updating the running average.

    Attributes:
        keys: Metric names tracked by this accumulator (set on first update).
        acc: Running averages aligned with ``keys``.
        key_counts: Per-metric sample counts aligned with ``keys``.
        device: Device used to build temporary reduction tensors.
        rank: Distributed rank.
        world_size: Number of distributed processes.
    """

    def __init__(
        self, device: Optional[torch.device] = None, keys: Optional[Sequence[str]] = None
    ) -> None:
        """Create a metric accumulator.

        Args:
            device: Device used for distributed metric reduction tensors. If
                ``None``, defaults to CPU behavior in ``torch.tensor``.
            keys: Optional canonical metric key list. If not provided, keys are
                initialized from the first ``update`` call (backward compatible).
        """
        self.keys: Optional[list[str]] = None
        self.acc: Optional[np.ndarray] = None
        self.key_counts: Optional[np.ndarray] = None
        self._key_to_idx: Optional[dict[str, int]] = None
        self.device = device
        # Cache distributed context once at construction.
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        self.rank = rank
        self.world_size = world_size

        if keys is not None:
            self.set_keys(keys)

    def set_keys(self, keys: Sequence[str]) -> None:
        """Set canonical metric keys and reset accumulators.

        Args:
            keys: Ordered metric names tracked by this accumulator.

        Raises:
            ValueError: If keys are empty or duplicated.
        """
        key_list = list(keys)
        if len(key_list) == 0:
            raise ValueError("keys must not be empty")
        if len(set(key_list)) != len(key_list):
            raise ValueError("keys must not contain duplicates")

        # Reset tracked schema and accumulators to the new canonical key order.
        self.keys = key_list
        self._key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.acc = np.zeros((len(self.keys),), dtype=np.float64)
        self.key_counts = np.zeros((len(self.keys),), dtype=np.float64)

    def reset(self) -> None:
        """Reset running averages and counters for configured keys."""
        if self.acc is not None:
            self.acc[:] = 0
        if self.key_counts is not None:
            self.key_counts[:] = 0

    def _get_reduce_device(self) -> torch.device:
        """Choose device for distributed reduction tensors."""
        if self.device is not None:
            return self.device
        # NCCL collectives require CUDA tensors.
        if self.world_size > 1 and dist.get_backend() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _init_keys_from_update(self, metrics: Mapping[str, MetricValue]) -> None:
        """Initialize canonical keys from update input (backward compatibility)."""
        if len(metrics) == 0:
            raise ValueError("Cannot initialize keys from an empty metrics mapping.")
        keys = list(metrics.keys())
        if self.world_size > 1:
            # Ensure every rank uses the same key schema/order.
            obj = [keys if self.rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            keys = obj[0]
        self.set_keys(keys)

    def _check_unknown_keys(self, metrics: Mapping[str, MetricValue]) -> None:
        """Validate that update metrics are part of configured keys.

        Args:
            metrics: Incoming metrics for current batch.

        Raises:
            ValueError: If unknown metric keys are provided.
        """
        assert self._key_to_idx is not None
        local_unknown = [k for k in metrics.keys() if k not in self._key_to_idx]
        has_unknown_local = 1 if len(local_unknown) > 0 else 0
        if self.world_size > 1:
            # Any rank reporting unknown keys should fail the whole update.
            flag = torch.tensor(
                [has_unknown_local], device=self._get_reduce_device(), dtype=torch.int64
            )
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            has_unknown = flag.item() > 0
        else:
            has_unknown = has_unknown_local > 0

        if has_unknown:
            unknown_str = ", ".join(local_unknown) if local_unknown else "<other rank>"
            raise ValueError(
                f"Unknown metric keys: {unknown_str}. "
                "Use set_keys(...) to redefine tracked keys."
            )

    def update(
        self, metrics: Mapping[str, MetricValue], num_samples: int = 1
    ) -> None:
        """Update running averages with a new batch of metrics.

        Uses a per-metric weighted-average update. Missing metrics in a batch are
        ignored (their counters are not incremented).

        For distributed training, batch metric sums and counts are reduced in the
        canonical key order and only rank 0 updates the accumulators.

            m^(i) = m^(i-1) + n^(i)/sum(n^(i)) (x^(i) - m^(i-1))

        where ``i`` is the batch index, ``x^(i)`` is the batch metric value,
        and ``n^(i)`` is the number of samples for that metric.

        Args:
            metrics: Scalar batch metrics keyed by name.
            num_samples: Number of samples represented by this batch.

        Raises:
            ValueError: If ``num_samples <= 0`` or unknown metric keys are found.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        if self.keys is None:
            self._init_keys_from_update(metrics)

        assert self.keys is not None
        assert self._key_to_idx is not None
        assert self.acc is not None
        assert self.key_counts is not None

        self._check_unknown_keys(metrics)

        dev = self._get_reduce_device()
        # Per-key batch aggregates in canonical key order.
        batch_sum = torch.zeros(len(self.keys), dtype=torch.float64, device=dev)
        batch_count = torch.zeros(len(self.keys), dtype=torch.float64, device=dev)

        bad_local = 0
        for k, v in metrics.items():
            idx = self._key_to_idx[k]
            value = float(v)
            if not math.isfinite(value):
                logging.warning("non-finite %s=%f", k, value)
                bad_local = 1
                continue
            batch_sum[idx] = value * num_samples
            batch_count[idx] = num_samples

        if self.world_size > 1:
            # Skip update globally if any rank produced non-finite values.
            bad_flag = torch.tensor([bad_local], device=dev, dtype=torch.int64)
            dist.all_reduce(bad_flag, op=dist.ReduceOp.SUM)
            if bad_flag.item() > 0:
                return
            # Reduce sums and counts in one collective.
            batch_stats = torch.stack((batch_sum, batch_count), dim=0)
            dist.reduce(batch_stats, dst=0, op=dist.ReduceOp.SUM)
            batch_sum = batch_stats[0]
            batch_count = batch_stats[1]
        elif bad_local:
            return

        # Only rank 0 owns the running averages.
        if self.rank != 0:
            return

        batch_sum_np = batch_sum.cpu().numpy()
        batch_count_np = batch_count.cpu().numpy()
        for i in range(len(self.keys)):
            n_i = batch_count_np[i]
            if n_i <= 0:
                # Missing metric for this update; keep previous average.
                continue
            x_i = batch_sum_np[i] / n_i
            new_count = self.key_counts[i] + n_i
            # Numerically stable incremental weighted-average update.
            r = n_i / new_count
            self.acc[i] += r * (x_i - self.acc[i])
            self.key_counts[i] = new_count

    @property
    def metrics(self) -> ODict[str, float]:
        """Return the accumulated metrics on rank 0.

        Returns:
            Ordered dictionary mapping metric names to running averages. Returns
            an empty ordered dictionary on non-zero ranks or before first update.
        """
        logs: ODict[str, float] = ODict()
        if self.rank != 0 or self.keys is None or self.acc is None:
            return logs
        for i, k in enumerate(self.keys):
            logs[k] = float(self.acc[i])

        return logs
