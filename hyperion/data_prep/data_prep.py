"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ActionYesNo
from pathlib import Path


class DataPrep:
    """Base class for data preparation classes.

    Attributes:
      corpus_dir: input data directory
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    registry = {}

    def __init__(self, corpus_dir, output_dir, use_kaldi_ids, target_sample_freq):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.use_kaldi_ids = use_kaldi_ids
        self.target_sample_freq = target_sample_freq

        self.output_dir.mkdir(exist_ok=True, parents=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.dataset_name()] = cls

    @staticmethod
    def dataset_name():
        raise NotImplementedError()

    @staticmethod
    def add_class_args(parser):
        parser.add_argument(
            "--corpus-dir", required=True, help="""input data directory""",
        )
        parser.add_argument(
            "--output-dir", required=True, help="""output data directory""",
        )
        parser.add_argument(
            "--use-kaldi-ids",
            default=False,
            action=ActionYesNo,
            help="""put speaker-id in front of segment id like kaldi""",
        )
        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help="""target sampling frequency to convert the audios to""",
        )
