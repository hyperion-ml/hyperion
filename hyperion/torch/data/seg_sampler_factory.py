"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Union, Optional
import logging
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo

from .audio_dataset import AudioDataset
from .feat_seq_dataset import FeatSeqDataset

from .seg_sampler import SegSampler
from .class_weighted_seg_chunk_sampler import ClassWeightedRandomSegChunkSampler
from .seg_chunk_sampler import SegChunkSampler
from .bucketing_seg_sampler import BucketingSegSampler

sampler_dict = {
    "class_weighted_random_seg_chunk_sampler": ClassWeightedRandomSegChunkSampler,
    "seg_sampler": SegSampler,
    "seg_chunk_sampler": SegChunkSampler,
    "bucketing_seg_sampler": BucketingSegSampler,
}


class SegSamplerFactory(object):
    """Factory class to create different types of samplers for
    sequencial data like audio or acoustic features.
    """

    @staticmethod
    def create(
        dataset: Union[AudioDataset, FeatSeqDataset],
        sampler_type: str = "class_weighted_random_seg_chunk_sampler",
        base_sampler_type: str = "seg_sampler",
        subbase_sampler_type: str = "seg_sampler",
        **kwargs,
    ):
        """Functions that creates a sequence sampler based on a dataset, sampler_type and sampler arguments.

        Args:
          dataset: sequence dataset object containing the data info of class AudioDataset or FeatSeqDataset.
          sampler_type: string indicating the sampler type.
        """

        sampler_class = sampler_dict[sampler_type]
        sampler_kwargs = sampler_class.filter_args(**kwargs)

        if sampler_type in ["bucketing_seg_sampler", "seg_chunk_sampler"]:
            base_sampler_class = sampler_dict[base_sampler_type]
            base_sampler_kwargs = base_sampler_class.filter_args(**kwargs)
            sampler_kwargs.update(base_sampler_kwargs)
            sampler_kwargs["base_sampler"] = base_sampler_class
            if base_sampler_type == "bucketing_seg_sampler":
                base_sampler_class = sampler_dict[subbase_sampler_type]
                base_sampler_kwargs = base_sampler_class.filter_args(**kwargs)
                sampler_kwargs.update(base_sampler_kwargs)

        if sampler_type in ["class_weighted_random_seg_chunk_sampler"]:
            try:
                class_name = sampler_kwargs["class_name"]
            except:
                class_name = "class_id"
            sampler_kwargs["class_info"] = dataset.class_info[class_name]

        logging.info(f"sampler-args={sampler_kwargs}")

        return sampler_class(dataset.seg_set, **sampler_kwargs)

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "sampler_type",
            "num_buckets",
            "min_chunk_length",
            "max_chunk_length",
            "min_batch_size",
            "max_batch_size",
            "max_batch_length",
            "num_chunks_per_seg_epoch",
            "num_segs_per_class",
            "num_chunks_per_seg",
            "weight_mode",
            "weight_exponent",
            "seg_weight_mode",
            "num_hard_prototypes",
            "class_name",
            "length_name",
            "iters_per_epoch",
            "batch_size",
            "shuffle",
            "drop_last",
            "seed",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--sampler-type",
            choices=sampler_dict.keys(),
            default="class_weighted_random_seg_chunk_sampler",
            help="batch sampler type",
        )

        parser.add_argument(
            "--base-sampler-type",
            choices=["seg_sampler", "bucketing_seg_sampler"],
            default="seg_sampler",
            help="base sampler used for seg_chunk_sampler or bucketing_seg_sampler",
        )

        parser.add_argument(
            "--min-chunk-length",
            type=float,
            default=4.0,
            help=("minimum length of the segment chunks"),
        )

        parser.add_argument(
            "--max-chunk-length",
            type=float,
            default=None,
            help=("maximum length of segment chunks"),
        )

        parser.add_argument(
            "--min-batch-size",
            type=int,
            default=64,
            help=("minimum batch size per gpu"),
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=None,
            help=(
                "maximum batch size per gpu, if None, estimated from max_batch_length"
            ),
        )

        parser.add_argument(
            "--batch-size",
            default=None,
            type=int,
            help=("deprecated, use min-batch-size instead"),
        )

        parser.add_argument(
            "--max-batch-duration",
            type=float,
            default=None,
            help=(
                "maximum accumlated duration of the batch, if None estimated from the min/max_batch_size and min/max_chunk_lengths"
            ),
        )

        parser.add_argument(
            "--iters-per-epoch",
            default=None,
            type=lambda x: x if (x == "auto" or x is None) else float(x),
            help=("deprecated, use --num-egs-per-seg-epoch instead"),
        )

        parser.add_argument(
            "--num-chunks-per-seg-epoch",
            default="auto",
            type=lambda x: x if x == "auto" else float(x),
            help=("number of times we sample a segment in each epoch"),
        )

        parser.add_argument(
            "--num-segs-per-class",
            type=int,
            default=1,
            help=("number of segments per class in batch"),
        )
        parser.add_argument(
            "--num-chunks-per-seg",
            type=int,
            default=1,
            help=("number of chunks per segment in batch"),
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
            "--seg-weight-mode",
            default="uniform",
            choices=["uniform", "data-prior"],
            help=("method to sample segments given a class"),
        )

        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help=("number of hard prototype classes per batch"),
        )

        parser.add_argument(
            "--drop-last",
            action=ActionYesNo,
            help="drops the last batch of the epoch",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="shuffles the segments or chunks at the beginning of the epoch",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help=("seed for sampler random number generator"),
        )

        parser.add_argument(
            "--length-name",
            default="duration",
            help="which column in the segment table indicates the duration of the segment",
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="which column in the segment table indicates the class of the segment",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
