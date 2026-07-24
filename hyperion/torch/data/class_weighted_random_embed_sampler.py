"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils import ClassInfo, InfoTable
from ...utils.misc import filter_func_args
from .hyper_sampler import HyperSampler


class ClassWeightedRandomEmbedSampler(HyperSampler):
    """
    Class-weighted sampler for fixed-dimensional embeddings.

    Each batch samples classes according to their configured weights and then
    samples a fixed number of embeddings from every selected class. Optional
    hard-prototype mining expands selected classes using a class affinity matrix.
    Embeddings are sampled with replacement, so a class may contribute the same
    embedding more than once in an epoch or batch.

    Attributes:
        embed_set (InfoTable): Embedding metadata containing ``id`` and the
            configured class column.
        class_info (ClassInfo): Class metadata and normalized sampling weights.
        batch_size (int): Number of embeddings per batch per rank, rounded up to
            a valid class-group size.
        avg_batch_size (int): Target average batch size used for epoch length.
        num_embeds_per_class (int): Number of embeddings sampled per class.
        weight_exponent (float): Exponent applied to class weights.
        weight_mode (str): Class weighting strategy.
        num_hard_prototypes (int): Number of classes represented per hard-prototype
            expansion, including the sampled class.
        hard_prototypes (Optional[torch.Tensor]): Hard-prototype class indices.
        class_name (str): Embedding metadata column containing class IDs.
        num_classes_per_batch (int): Number of base classes sampled per batch.
        batch (int): Number of batches yielded in the current epoch.
        map_class_idx_to_ids (Any): Mapping from class indices to class IDs.
        map_class_to_embed_idx (Dict[Any, np.ndarray]): Embedding row indices per
            class ID.
        _len (int): Number of batches yielded per epoch and rank.
    """

    def __init__(
        self,
        embed_set: InfoTable,
        class_info: ClassInfo,
        batch_size: int = 1,
        num_embeds_per_class: int = 1,
        weight_exponent: float = 1.0,
        weight_mode: str = "custom",
        num_hard_prototypes: int = 0,
        affinity_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        class_name: str = "class_id",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
    ) -> None:
        """
        Initialize the class-weighted embedding sampler.

        Args:
            embed_set: Embedding metadata containing ``id`` and ``class_name``.
            class_info: Class metadata containing class IDs, indices, and weights.
            batch_size: Requested number of embeddings per batch per rank. It is
                rounded up to a multiple of the class-group size.
            num_embeds_per_class: Number of embeddings sampled for each selected
                class, with replacement.
            weight_exponent: Exponent applied to class weights.
            weight_mode: Class weighting strategy: ``custom``, ``uniform``, or
                ``data-prior``. Data-prior weights use the number of embeddings
                belonging to each class.
            num_hard_prototypes: Number of classes represented for each selected
                class when hard-prototype mining is enabled.
            affinity_matrix: Optional class affinity matrix for hard-prototype
                mining.
            class_name: Column in ``embed_set`` containing class IDs.
            max_batches_per_epoch: Optional maximum number of batches per epoch.
            shuffle: Whether to vary the random seed by epoch.
            seed: Base random seed.
        """
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=shuffle, seed=seed
        )
        if len(embed_set) == 0:
            raise ValueError("embed_set must contain at least one row.")
        if len(class_info) == 0:
            raise ValueError("class_info must contain at least one row.")
        if "id" not in embed_set or class_name not in embed_set:
            raise ValueError(
                f"embed_set must contain 'id' and {class_name!r} columns."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if num_embeds_per_class <= 0:
            raise ValueError(
                "num_embeds_per_class must be positive, "
                f"got {num_embeds_per_class}."
            )
        if num_hard_prototypes < 0:
            raise ValueError(
                "num_hard_prototypes must be non-negative, "
                f"got {num_hard_prototypes}."
            )
        if weight_mode not in ("custom", "uniform", "data-prior"):
            raise ValueError(
                "weight_mode must be one of 'custom', 'uniform', or 'data-prior', "
                f"got {weight_mode}."
            )

        self.embed_set = embed_set
        self.class_info = class_info
        self.batch_size = batch_size
        self.avg_batch_size = batch_size
        self.num_embeds_per_class = num_embeds_per_class
        self.weight_exponent = weight_exponent
        self.weight_mode = weight_mode
        self.num_hard_prototypes = num_hard_prototypes
        self.class_name = class_name
        self.batch = 0

        self._gather_class_info()
        self._set_class_weights()
        self.set_hard_prototypes(affinity_matrix)
        self._round_batch_size()
        self._compute_num_classes_per_batch()
        self._compute_len()

        if self._len == 0:
            raise ValueError(
                "ClassWeightedRandomEmbedSampler would yield zero batches per "
                "rank. Increase the number of embeddings or reduce world_size."
            )

        logging.info(
            "sampler batches/epoch=%d batch-size=%d classes/batch=%d "
            "embeds/class=%d",
            self._len,
            self.batch_size,
            self.num_classes_per_batch,
            self.num_embeds_per_class,
        )

    def _set_seed(self) -> None:
        """Initialize the random generator for the current epoch and rank."""
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.rank)
        else:
            self.rng.manual_seed(self.seed + 100 * self.rank)

    def _compute_len(self) -> None:
        """Compute the number of batches yielded by each rank per epoch."""
        self._len = int(
            math.ceil(len(self.embed_set) / self.avg_batch_size / self.world_size)
        )
        if self.max_batches_per_epoch is not None:
            self._len = min(self._len, self.max_batches_per_epoch)

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        Returns:
            Number of batches yielded by this rank.
        """
        return self._len

    def _gather_class_info(self) -> None:
        """Build mappings from class IDs to embedding row indices."""
        logging.info("Creating class-to-embedding index mapping")
        self.map_class_idx_to_ids = self.class_info.df[["class_idx", "id"]]
        self.map_class_idx_to_ids.set_index("class_idx", inplace=True)

        map_class_to_embeds = self.embed_set.df[["id", self.class_name]].set_index(
            self.class_name
        )
        self.map_class_to_embed_idx: Dict[Any, np.ndarray] = {}
        self._classes_without_embeds: List[Any] = []
        self.class_info["num_embeds"] = 0

        for class_id in self.class_info["id"].values:
            if class_id not in map_class_to_embeds.index:
                self._classes_without_embeds.append(class_id)
                continue

            embed_ids = map_class_to_embeds.loc[class_id, "id"]
            if np.isscalar(embed_ids):
                embed_ids = [embed_ids]
            else:
                embed_ids = embed_ids.values
            embed_idx = np.asarray(self.embed_set.get_loc(embed_ids), dtype=int)
            self.map_class_to_embed_idx[class_id] = embed_idx
            self.class_info.loc[class_id, "num_embeds"] = len(embed_idx)

        for class_id in self._classes_without_embeds:
            self.map_class_to_embed_idx[class_id] = np.asarray([], dtype=int)

    def _set_class_weights(self) -> None:
        """Set, exponentiate, and validate the effective class weights."""
        if self.weight_mode == "uniform":
            self.class_info.set_uniform_weights()
        elif self.weight_mode == "data-prior":
            self.class_info.set_weights(self.class_info["num_embeds"].values)

        weights = np.asarray(self.class_info["weights"].values, dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Class weights must be finite and non-negative.")
        if self.weight_exponent != 1.0:
            self.class_info.exp_weights(self.weight_exponent)

        if self._classes_without_embeds:
            self.class_info.loc[self._classes_without_embeds, "weights"] = 0.0
            try:
                self.class_info.renorm_weights()
            except ValueError as error:
                raise ValueError("No classes in class_info have embeddings.") from error

        if self.class_info["weights"].sum() <= 0:
            raise ValueError("At least one class must have a positive weight.")

    @property
    def hard_prototype_mining(self) -> bool:
        """
        Whether hard-prototype mining is active.

        Returns:
            ``True`` when multiple prototypes were requested and initialized.
        """
        return self.num_hard_prototypes > 1 and self.hard_prototypes is not None

    def set_hard_prototypes(
        self, affinity_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> None:
        """
        Initialize hard-prototype class indices from an affinity matrix.

        Args:
            affinity_matrix: Square class affinity matrix. ``None`` disables
                hard-prototype mining.
        """
        if affinity_matrix is None or self.num_hard_prototypes <= 1:
            self.hard_prototypes = None
            return

        affinity = torch.as_tensor(affinity_matrix).clone()
        if affinity.ndim != 2 or affinity.size(0) != affinity.size(1):
            raise ValueError(
                "affinity_matrix must be square, "
                f"got shape={tuple(affinity.shape)}."
            )
        class_idx = self.class_info["class_idx"].values
        if np.any(class_idx < 0) or np.any(class_idx >= affinity.size(0)):
            raise ValueError(
                "class_info class_idx values must index affinity_matrix, "
                f"whose shape is {tuple(affinity.shape)}."
            )
        if self.num_hard_prototypes > affinity.size(1):
            raise ValueError(
                "num_hard_prototypes cannot exceed affinity_matrix size, "
                f"got {self.num_hard_prototypes} for {affinity.size(1)} classes."
            )

        zero_weight = self.class_info["weights"].values == 0
        num_active_classes = int((~zero_weight).sum())
        if self.num_hard_prototypes > num_active_classes:
            raise ValueError(
                "num_hard_prototypes cannot exceed the number of sampleable "
                f"classes, got {self.num_hard_prototypes} for {num_active_classes}."
            )
        if np.any(zero_weight):
            zero_idx = self.class_info.loc[zero_weight, "class_idx"].values
            affinity[:, zero_idx] = -1000
        for class_idx_i in range(affinity.size(1)):
            if not np.any(self.class_info["class_idx"].values == class_idx_i):
                affinity[:, class_idx_i] = -1000

        self.hard_prototypes = torch.topk(
            affinity, self.num_hard_prototypes, dim=-1
        ).indices

    def get_hard_prototypes(self, class_idx: Union[int, np.ndarray]) -> np.ndarray:
        """
        Return hard-prototype class indices for the requested classes.

        Args:
            class_idx: Scalar or array of class indices.

        Returns:
            Flattened array of hard-prototype class indices.
        """
        class_idx_np = np.asarray(class_idx)
        class_idx_t = torch.as_tensor(class_idx_np, dtype=torch.long)
        return self.hard_prototypes[class_idx_t].cpu().numpy().flatten()

    def _compute_num_classes_per_batch(self) -> None:
        """Compute the number of base classes sampled in each batch."""
        num_classes = self.batch_size / self.num_embeds_per_class
        if self.hard_prototype_mining:
            num_classes /= self.num_hard_prototypes
        self.num_classes_per_batch = max(1, int(math.ceil(num_classes)))

    def _round_batch_size(self) -> None:
        """Round the requested batch size up to a complete class-group size."""
        class_group_size = self.num_embeds_per_class
        if self.hard_prototype_mining:
            class_group_size *= self.num_hard_prototypes
        if self.batch_size % class_group_size == 0:
            return

        rounded_batch_size = (
            math.ceil(self.batch_size / class_group_size) * class_group_size
        )
        logging.info(
            "Rounding batch_size from %d to %d to fit class groups of %d embeddings.",
            self.batch_size,
            rounded_batch_size,
            class_group_size,
        )
        self.batch_size = rounded_batch_size
        self.avg_batch_size = rounded_batch_size

    def _get_class_weights(self) -> torch.Tensor:
        """
        Return normalized class sampling weights.

        Returns:
            Tensor containing one probability per class.
        """
        weights = np.asarray(self.class_info["weights"].values, dtype=float)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Class sampling weights must have a finite positive sum.")
        return torch.as_tensor(weights / total, dtype=torch.float32)

    def _sample_classes(self, num_classes: int) -> np.ndarray:
        """
        Sample class IDs according to the configured class weights.

        Args:
            num_classes: Number of base class IDs to sample.

        Returns:
            Selected class IDs, expanded with hard prototypes when enabled.
        """
        row_idx = torch.multinomial(
            self._get_class_weights(),
            num_samples=num_classes,
            replacement=True,
            generator=self.rng,
        ).numpy()
        id_col_idx = self.class_info.get_col_idx("id")
        class_ids = self.class_info.iloc[row_idx, id_col_idx].values
        if self.hard_prototype_mining:
            class_idx = self.class_info.loc[class_ids, "class_idx"].values
            hard_class_idx = self.get_hard_prototypes(class_idx)
            class_ids = self.map_class_idx_to_ids.loc[hard_class_idx, "id"].values
        return class_ids

    def _sample_embeds(self, class_ids: np.ndarray) -> List[str]:
        """
        Sample embedding IDs for each selected class.

        Args:
            class_ids: Class IDs selected for the batch.

        Returns:
            Embedding IDs sampled with replacement.
        """
        id_col_idx = self.embed_set.get_col_idx("id")
        embed_ids: List[str] = []
        for class_id in class_ids:
            embed_idx = self.map_class_to_embed_idx[class_id]
            if len(embed_idx) == 0:
                raise ValueError(f"No embeddings found for class={class_id}.")
            selected = torch.randint(
                low=0,
                high=len(embed_idx),
                size=(self.num_embeds_per_class,),
                generator=self.rng,
            ).numpy()
            selected_idx = embed_idx[selected]
            embed_ids.extend(self.embed_set.iloc[selected_idx, id_col_idx].tolist())
        return embed_ids

    def __next__(self) -> List[str]:
        """
        Sample and return the next embedding-ID batch.

        Returns:
            Embedding IDs selected for the next batch.

        Raises:
            StopIteration: When all batches in the epoch have been yielded.
        """
        if self.batch >= self._len:
            raise StopIteration

        profile_next = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
        if profile_next:
            start_time = time.perf_counter()
        class_ids = self._sample_classes(self.num_classes_per_batch)
        embed_ids = self._sample_embeds(class_ids)
        if profile_next:
            logging.debug(
                "%s batch=%d classes=%d embeddings=%d total-ms=%.3f",
                __name__,
                self.batch,
                len(class_ids),
                len(embed_ids),
                (time.perf_counter() - start_time) * 1e3,
            )
        if self.batch == 0:
            logging.info("batch 0 embed_ids=%s", str(embed_ids[:10]))

        self.batch += 1
        return embed_ids

    def __iter__(self) -> "ClassWeightedRandomEmbedSampler":
        """
        Initialize iteration and replay a requested resume offset.

        Returns:
            This sampler instance.
        """
        resume_batch = self.init_batch
        self.init_batch = 0
        super().__iter__()
        if resume_batch != 0:
            logging.info(
                "Replaying %d batches to resume ClassWeightedRandomEmbedSampler.",
                resume_batch,
            )
            for _ in range(min(resume_batch, self._len)):
                try:
                    next(self)
                except StopIteration:
                    break
        return self

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter keyword arguments accepted by the constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Constructor-compatible keyword arguments.
        """
        return filter_func_args(ClassWeightedRandomEmbedSampler.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """
        Add class-weighted embedding sampler arguments to a parser.

        Args:
            parser: Parser to extend.
            prefix: Optional nested parser prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Target number of embeddings per batch per GPU.",
        )
        parser.add_argument(
            "--num-embeds-per-class",
            type=int,
            default=1,
            help="Number of embeddings sampled per selected class.",
        )
        parser.add_argument(
            "--weight-exponent",
            type=float,
            default=1.0,
            help="Exponent applied to class sampling weights.",
        )
        parser.add_argument(
            "--weight-mode",
            choices=["custom", "uniform", "data-prior"],
            default="custom",
            help="Class weighting strategy.",
        )
        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help="Number of hard-prototype classes per selected class.",
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
            help="Vary the sampling seed by epoch.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Base random seed.",
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="Embedding metadata column containing class IDs.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
