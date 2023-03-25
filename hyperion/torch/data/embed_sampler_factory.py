"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Optional, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .bucketing_seg_sampler import BucketingSegSampler
from .class_weighted_embed_sampler import ClassWeightedEmbedSampler
from .embed_dataset import EmbedDataset
from .embed_sampler import EmbedSampler

sampler_dict = {
    "class_weighted_embed_sampler": ClassWeightedEmbedSampler,
    "embed_sampler": EmbedSampler,
}


class EmbedSamplerFactory(object):
    """Factory class to create different types of samplers for
    embeddings like x-vectors.
    """

    @staticmethod
    def create(
        dataset: EmbedDataset,
        sampler_type: str = "class_weighted_embed_sampler",
        **kwargs,
    ):
        """Functions that creates a sampler based on a dataset, sampler_type and sampler arguments.

        Args:
          dataset: embeddings dataset object containing the data info 
          sampler_type: string indicating the sampler type.
        """

        sampler_class = sampler_dict[sampler_type]
        sampler_kwargs = sampler_class.filter_args(**kwargs)

        if sampler_type in ["class_weighted_embed_sampler"]:
            try:
                class_name = sampler_kwargs["class_name"]
            except:
                class_name = "class_id"
            sampler_kwargs["class_info"] = dataset.class_info[class_name]

        logging.info(f"sampler-args={sampler_kwargs}")

        return sampler_class(dataset.embed_info, **sampler_kwargs)

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "batch_size",
            "num_embeds_per_class",
            "weight_exponent",
            "weight_mode",
            "num_hard_prototypes",
            "class_name",
            "shuffle",
            "seed",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--batch-size", type=int, default=1, help=("batch size per gpu"),
        )

        parser.add_argument(
            "--num-embeds-per-class",
            type=int,
            default=1,
            help=("number of embeds per class in batch"),
        )
        parser.add_argument(
            "--weight-exponent",
            default=1.0,
            type=float,
            help=("exponent for class weights"),
        )
        parser.add_argument(
            "--weight-mode",
            default="custom",
            choices=["custom", "uniform", "data-prior"],
            help=("method to get the class weights"),
        )

        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help=("number of hard prototype classes per batch"),
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="shuffles the embeddings at the beginning of the epoch",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help=("seed for sampler random number generator"),
        )

        parser.add_argument(
            "--class-name",
            default="class_id",
            help="which column in the info table indicates the class",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
