"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Type

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .class_weighted_random_embed_sampler import ClassWeightedRandomEmbedSampler
from .embed_dataset import EmbedDataset
from .embed_sampler import EmbedSampler
from .hyper_sampler import HyperSampler

sampler_dict: Dict[str, Type[HyperSampler]] = {
    "class_weighted_random_embed_sampler": ClassWeightedRandomEmbedSampler,
    "embed_sampler": EmbedSampler,
}


class EmbedSamplerFactory:
    """Create samplers for fixed-dimensional embedding datasets.

    Attributes:
        sampler_types (Dict[str, Type[HyperSampler]]): Mapping from sampler type
            names to their implementation classes.
    """

    sampler_types: Dict[str, Type[HyperSampler]] = sampler_dict

    @staticmethod
    def create(
        dataset: EmbedDataset,
        sampler_type: str = "class_weighted_random_embed_sampler",
        **kwargs: Any,
    ) -> HyperSampler:
        """Create an embedding sampler from dataset metadata and configuration.

        Args:
            dataset: Dataset providing ``embed_info`` and, for class-weighted
                sampling, a mapping of class names to ``ClassInfo`` tables.
            sampler_type: Registered sampler type to instantiate.
            **kwargs: Sampler configuration arguments.

        Returns:
            Configured embedding sampler instance.

        Raises:
            ValueError: If ``sampler_type`` is unknown or the requested class
                information is unavailable from the dataset.
        """
        if sampler_type not in EmbedSamplerFactory.sampler_types:
            raise ValueError(f"Unknown sampler_type={sampler_type}.")

        sampler_class = EmbedSamplerFactory.sampler_types[sampler_type]
        sampler_kwargs = sampler_class.filter_args(**kwargs)

        if sampler_type == "class_weighted_random_embed_sampler":
            class_name = sampler_kwargs.get("class_name", "class_id")
            try:
                sampler_kwargs["class_info"] = dataset.class_info[class_name]
            except (AttributeError, KeyError, TypeError) as error:
                raise ValueError(
                    "Dataset does not provide class_info for "
                    f"class_name={class_name!r}."
                ) from error

        logging.info("sampler-args=%s", sampler_kwargs)
        return sampler_class(dataset.embed_info, **sampler_kwargs)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments accepted by :meth:`create`.

        Args:
            **kwargs: Candidate factory and sampler configuration arguments.

        Returns:
            Dictionary containing only supported factory and sampler arguments.
        """
        valid_args = (
            "dataset",
            "sampler_type",
            "batch_size",
            "num_embeds_per_class",
            "weight_exponent",
            "weight_mode",
            "num_hard_prototypes",
            "affinity_matrix",
            "class_name",
            "max_batches_per_epoch",
            "shuffle",
            "drop_last",
            "seed",
        )
        return {key: kwargs[key] for key in valid_args if key in kwargs}

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Add embedding sampler factory arguments to a parser.

        Args:
            parser: Argument parser to populate.
            prefix: Optional key under which to nest these arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--sampler-type",
            choices=EmbedSamplerFactory.sampler_types.keys(),
            default="class_weighted_random_embed_sampler",
            help="Embedding sampler implementation to use.",
        )
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
            help="Class weighting strategy for class-weighted sampling.",
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
            "--drop-last",
            action=ActionYesNo,
            help="Drop embeddings that cannot fill a complete distributed batch.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Base random seed for reproducible sampling.",
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="Embedding metadata column containing class IDs.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
