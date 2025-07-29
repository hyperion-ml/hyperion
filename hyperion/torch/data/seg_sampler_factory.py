"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Optional, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .audio_dataset import AudioDataset
from .bucketing_seg_sampler import BucketingSegSampler
from .class_weighted_seg_chunk_sampler import ClassWeightedRandomSegChunkSampler
from .feat_seq_dataset import FeatSeqDataset
from .legacy_audio_dataset import LegacyAudioDataset
from .seg_chunk_sampler import SegChunkSampler
from .seg_sampler import LengthSamplingMethod, SegSampler

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
        dataset: Union[AudioDataset, LegacyAudioDataset, FeatSeqDataset],
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
        if "batch_size" in kwargs:
            if kwargs["batch_size"] is not None:
                kwargs["min_batch_size"] = kwargs.pop("batch_size")
            else:
                del kwargs["batch_size"]

        sampler_class = sampler_dict[sampler_type]
        sampler_kwargs = sampler_class.filter_args(**kwargs)

        if base_sampler_type not in sampler_dict:
            raise ValueError(f"Unknown base_sampler_type={base_sampler_type}")

        if subbase_sampler_type not in sampler_dict:
            raise ValueError(f"Unknown subbase_sampler_type={subbase_sampler_type}")

        if sampler_type in ["bucketing_seg_sampler", "seg_chunk_sampler"]:
            base_sampler_class = sampler_dict[base_sampler_type]
            base_sampler_kwargs = base_sampler_class.filter_args(**kwargs)
            sampler_kwargs.update(base_sampler_kwargs)
            sampler_kwargs["base_sampler"] = base_sampler_class
            if base_sampler_type == "bucketing_seg_sampler":
                subbase_sampler_class = sampler_dict[subbase_sampler_type]
                subbase_sampler_kwargs = subbase_sampler_class.filter_args(**kwargs)
                sampler_kwargs.update(subbase_sampler_kwargs)

        if sampler_type in ["class_weighted_random_seg_chunk_sampler"]:
            try:
                class_name = sampler_kwargs["class_name"]
            except:
                class_name = "class_id"
            sampler_kwargs["class_info"] = dataset.class_info[class_name]

        logging.info(f"sampler-args={sampler_kwargs}")

        return sampler_class(dataset.segments, **sampler_kwargs)

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "dataset",
            "sampler_type",
            "base_sampler_type",
            "subbase_sampler_type",
            "num_buckets",
            "min_chunk_length",
            "max_chunk_length",
            "min_chunk_overlap",
            "max_chunk_overlap",
            "length_sampling_method",
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
            "max_batches_per_epoch",
            "shuffle",
            "drop_last",
            "sort_by_length",
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
            help="Type of sampler to use. Determines the batching and sampling strategy for segment or chunk data.",
        )

        parser.add_argument(
            "--base-sampler-type",
            choices=["seg_sampler", "bucketing_seg_sampler"],
            default="seg_sampler",
            help="Base sampler class used by samplers like seg_chunk_sampler or bucketing_seg_sampler to draw batches.",
        )
        parser.add_argument(
            "--subbase-sampler-type",
            choices=["seg_sampler", "bucketing_seg_sampler"],
            default="seg_sampler",
            help="Sampler used as a base within a bucketing sampler (used when base-sampler-type is bucketing_seg_sampler).",
        )

        parser.add_argument(
            "--num-buckets",
            type=int,
            default=10,
            help="Number of buckets to divide the dataset into by segment length (for bucketing samplers).",
        )

        parser.add_argument(
            "--min-chunk-length",
            type=float,
            default=4.0,
            help="Minimum chunk duration in seconds when slicing segments into chunks.",
        )

        parser.add_argument(
            "--max-chunk-length",
            type=float,
            default=None,
            help="Maximum chunk duration in seconds. If not set, equals min-chunk-length.",
        )

        parser.add_argument(
            "--min-chunk-overlap",
            type=float,
            default=0.0,
            help="Minimum overlap in seconds between consecutive chunks extracted from a segment.",
        )
        parser.add_argument(
            "--max-chunk-overlap",
            type=float,
            default=None,
            help="Maximum overlap in seconds between chunks. If None, uses min-chunk-overlap.",
        )
        parser.add_argument(
            "--length-sampling-method",
            choices=LengthSamplingMethod.choices(),
            default=LengthSamplingMethod.UNIFORM.value,
            help="Strategy for sampling chunk lengths. 'uniform' draws from a range, 'maximum' always uses max-chunk-length.",
        )

        parser.add_argument(
            "--min-batch-size",
            type=int,
            default=1,
            help="Minimum number of samples (segments/chunks) in a batch per GPU.",
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=None,
            help="Maximum batch size per GPU. If None, it will be estimated based on max-batch-length.",
        )

        parser.add_argument(
            "--batch-size",
            default=None,
            type=int,
            help="(Deprecated) Use --min-batch-size instead. Sets fixed batch size if provided.",
        )

        parser.add_argument(
            "--max-batch-length",
            type=float,
            default=None,
            help="Maximum total duration (in seconds) of segments/chunks in a batch. Used to control memory usage.",
        )

        parser.add_argument(
            "--iters-per-epoch",
            default=None,
            type=lambda x: x if (x == "auto" or x is None) else float(x),
            help="(Deprecated) Use --num-chunks-per-seg-epoch instead. Number of iterations per epoch.",
        )

        parser.add_argument(
            "--num-chunks-per-seg-epoch",
            default="auto",
            type=lambda x: x if x == "auto" else float(x),
            help="How many chunks to draw from each segment per epoch. Can be an int or 'auto'.",
        )

        parser.add_argument(
            "--num-segs-per-class",
            type=int,
            default=1,
            help="Number of segments to sample per class when forming a batch (used in class-weighted samplers).",
        )
        parser.add_argument(
            "--num-chunks-per-seg",
            type=int,
            default=1,
            help="Number of chunks to extract per segment in a single batch.",
        )

        parser.add_argument(
            "--weight-exponent",
            default=1.0,
            type=float,
            help="Exponent to apply when transforming class weights (e.g., for power-law reweighting).",
        )
        parser.add_argument(
            "--weight-mode",
            default="custom",
            choices=["custom", "uniform", "data-prior"],
            help="How to assign weights to classes. 'data-prior' uses segment durations, 'uniform' uses equal weights.",
        )

        parser.add_argument(
            "--seg-weight-mode",
            default="uniform",
            choices=["uniform", "data-prior"],
            help="How to sample segments within a class. 'uniform' is equal probability, 'data-prior' uses durations.",
        )

        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help="Number of hard prototype classes to sample based on similarity (if affinity matrix is provided).",
        )

        parser.add_argument(
            "--drop-last",
            action=ActionYesNo,
            help="If set, drops the last batch in the epoch if it has fewer than min-batch-size samples.",
        )

        parser.add_argument(
            "--max-batches-per-epoch",
            type=int,
            default=None,
            help="Optional limit on number of batches per epoch (across all samplers).",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="Shuffle segment order at the start of each epoch.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Random seed for deterministic sampling across epochs and distributed workers.",
        )

        parser.add_argument(
            "--length-name",
            default="duration",
            help="Column name in the segment table that represents the segment's duration (in seconds).",
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="Column name in the segment table that represents the class or label of each segment.",
        )
        parser.add_argument(
            "--skip-long-segs",
            default=False,
            action=ActionYesNo,
            help="If True, skips segments longer than max-batch-length when sampling batches. This is useful to avoid memory issues with long segments."
            "",
        )

        parser.add_argument(
            "--sort-by-length",
            default=True,
            action=ActionYesNo,
            help="If True, sorts batch items by duration (descending) to improve padding efficiency.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
