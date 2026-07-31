"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo, ArgumentParser
from tqdm import tqdm

# from ..utils import ClassInfo, HyperDataset, RecordingSet, SegmentSet, TrialKey, TrialNdx
from ..utils.misc import PathLike
from .data_prep import DataPrep


class HFDatasetDataPrep(DataPrep):
    """
    Prepares a Hugging Face dataset by downloading and extracting it into a corpus directory,
    and then generating metadata and segment tables for downstream processing.

    Attributes:
        hf_data_path (str | Path | None): Hugging Face dataset path or ID.
        corpus_dir (PathLike): Directory where audio files are extracted.
        config (str | None): Dataset configuration name (if applicable).
        split (str | None): Dataset split (e.g., 'train', 'test').
        output_dir (PathLike): Output directory for processed metadata.
        use_kaldi_ids (bool): Whether to use Kaldi-style IDs.
        target_sample_freq (Optional[int]): Desired sample rate for output audio.
        force_download (bool): Whether to re-download even if metadata exists.
        cache_dir (Optional[str]): Hugging Face cache directory.
    """

    def __init__(
        self,
        hf_data_path: Union[PathLike, None],
        corpus_dir: PathLike,
        config: Union[str, None],
        split: Union[str, None],
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the HFDatasetDataPrep class.
        """
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.hf_data_path = hf_data_path
        self.config = config
        self.split = split
        self.force_download = force_download
        self.cache_dir = cache_dir

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset identifier string."""
        return "hf_dataset"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """Add command-line arguments for Hugging Face dataset preparation."""
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

    def do_i_download_corpus(self) -> bool:
        """
        Determines whether the dataset should be downloaded again.

        Returns:
            bool: True if download is needed, False otherwise.
        """
        meta_file = self.corpus_dir / "metadata.csv"
        return not meta_file.exists() or self.force_download

    def download_corpus(self) -> None:
        """
        Downloads and extracts all splits/configs of a Hugging Face dataset into structured format.

        Creates a metadata CSV in the corpus_dir.
        """

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

    def _prepare_from_meta(self, df_meta: pd.DataFrame) -> None:
        """
        Processes metadata and creates segment and recording tables.

        Args:
            df_meta (pd.DataFrame): DataFrame containing metadata information.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def prepare(self) -> None:
        """
        Orchestrates the download and processing of a Hugging Face dataset into segment/recording tables.
        """
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

    def extract_hf_item(item: Dict[str, Any], extract_dir: PathLike) -> Dict:
        """
        Must be implemented in subclass: logic to extract a single HF row into a local audio file.

        Args:
            row (Dict): A single row from the Hugging Face dataset.
            extract_dir (Path): Directory where audio and metadata should be saved.

        Returns:
            Dict: Extracted metadata including audio filename, speaker, etc.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")
