"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

# import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

# from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet, TrialKey, TrialNdx
from ..utils.misc import PathLike
from .data_prep import DataPrep


class HFDatasetDataPrep(DataPrep):
    """Base Class for preparing Hugging Face database into tables,

    Attributes:
      hf_data_path: Hugging Face data path or corpus id
      corpus_dir: data directory where audios are extracted
      config: hf config names
      split: train/dev/eval
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
      force_download: download the data again even if corpus dir exist
      cache_dir: Directory to read/write data. Defaults to ~/.cache/huggingface/datasets
    """

    def __init__(
        self,
        hf_data_path: Union[PathLike, None],
        corpus_dir: PathLike,
        config: Union[str, None],
        split: Union[str, None],
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.hf_data_path = hf_data_path
        self.config = config
        self.split = split
        self.force_download = force_download
        self.cache_dir = cache_dir

    @staticmethod
    def dataset_name():
        return "hf_dataset"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--hf-data-path",
            default=None,
            help="Hugging Face data path or corpus id",
        )
        # parser.add_argument(
        #     "--config",
        #     default=None,
        #     help="""hf dataset configuration name""",
        # )
        # parser.add_argument(
        #     "--split",
        #     choices=[
        #         "train",
        #         "dev",
        #         "eval",
        #     ],
        #     help="""if we prepare the data for ["train", "dev", "eval"]""",
        #     required=True,
        # )
        parser.add_argument(
            "--force-download",
            default=False,
            action=ActionYesNo,
            help="download the data again even if corpus dir exist",
        )
        parser.add_argument(
            "--cache-dir",
            default=None,
            help="Directory to read/write data. Defaults to ~/.cache/huggingface/datasets",
        )

    def do_i_download_corpus(self):
        meta_file = self.corpus_dir / "metadata.csv"
        return not meta_file.exists() or self.force_download

    def download_corpus(self):

        from datasets import (
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
            load_dataset_builder,
        )

        logging.info("Downloading and extracting corpus to audios")
        # ds_builder = load_dataset_builder(self.hf_data_path)

        # print(ds_builder.info.description)
        # print(ds_builder.info.features)
        avail_configs = get_dataset_config_names(self.hf_data_path)
        avail_splits = get_dataset_split_names(self.hf_data_path)

        items = []
        for config in avail_configs:
            for split in avail_splits:
                logging.info(f"Extracting {config=} {split=}")
                data = load_dataset(
                    self.hf_data_path,
                    name=config,
                    split=split,
                    cache_dir=self.cache_dir,
                    streaming=True,
                )
                extract_dir = self.corpus_dir
                if config != "default":
                    extract_dir = extract_dir / config

                extract_dir = extract_dir / split
                extract_dir.mkdir(exist_ok=True, parents=True)

                for row in tqdm(data):
                    item = self.extract_hf_item(row, extract_dir)
                    item["config"] = config
                    item["split"] = split
                    items.append(item)

        output_file = self.corpus_dir / "metadata.csv"
        df = pd.DataFrame(items)
        df.to_csv(output_file, sep=",", index=False)

    def prepare(self):
        logging.info(
            "Peparing %s Dataset %s %s %s -> corpus_dir:%s -> data_dir:%s",
            self.dataset_name(),
            self.hf_data_path if self.hf_data_path is not None else "",
            str(self.config) if self.config else "",
            self.split,
            self.corpus_dir,
            self.output_dir,
        )

        if self.do_i_download_corpus():
            assert self.hf_data_path is not None
            self.download_corpus()

        meta_file = self.corpus_dir / "metadata.csv"
        df_meta = pd.read_csv(meta_file, sep=",")
        self._prepare_from_meta(df_meta)
