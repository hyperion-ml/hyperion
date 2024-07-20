"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

# import numpy as np
import pandas as pd
import soundfile as sf
from jsonargparse import ActionYesNo

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .hf_dataset import HFDatasetDataPrep


class FakeCodecDataPrep(HFDatasetDataPrep):
    """Class for preparing Fake Codec database into tables,

    Attributes:
      hf_data_path: Hugging Face data path or corpus id
      corpus_dir: data directory where audios are extracted
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
      force_download: download the data again even if corpus dir exist
    """

    def __init__(
        self,
        hf_data_path: Union[PathLike, None],
        corpus_dir: PathLike,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            hf_data_path,
            corpus_dir,
            config=None,
            split="train",
            output_dir=output_dir,
            use_kaldi_ids=False,
            target_sample_freq=target_sample_freq,
            num_threads=num_threads,
            force_download=force_download,
            cache_dir=cache_dir,
        )

    @staticmethod
    def dataset_name():
        return "fake_codec"

    @staticmethod
    def add_class_args(parser):
        HFDatasetDataPrep.add_class_args(parser)

    def extract_hf_item(self, item: Dict[str, Any], extract_dir: PathLike):
        # print(item)
        seg_id = Path(item["audio"]["path"]).with_suffix("")
        audio_dir = extract_dir / "audio"
        if not audio_dir.is_dir():
            audio_dir.mkdir(parents=True)
        storage_path = str(audio_dir / Path(item["audio"]["path"]).with_suffix(".flac"))
        storage_path_suffix = re.sub(str(self.corpus_dir), "", storage_path)[1:]
        spoof_det = "spoof" if item["label"] == "spoofing" else "bonafide"
        spoof_access = "LA" if item["label"] == "spoofing" else None
        audio = item["audio"]["array"]
        fs = item["audio"]["sampling_rate"]
        duration = len(audio) / fs
        output_item = {
            "id": seg_id,
            "storage_path": storage_path_suffix,
            "speaker": item["speaker_id"],
            "codec": item["codec_name"],
            "sample_freq": fs,
            "spoof_det": spoof_det,
            "language": "english",
            "spoof_access": spoof_access,
            "duration": duration,
        }
        sf.write(storage_path, audio, samplerate=fs)
        return output_item

    def _prepare_from_meta(self, df_meta: pd.DataFrame):

        logging.info("making SegmentsSet")
        df_segs = df_meta.drop(["storage_path", "sample_freq"], axis=1)
        segments = SegmentSet(df_segs)

        logging.info("making RecordingSet")
        df_recs = df_meta[["id", "storage_path", "duration", "sample_freq"]]
        if self.target_sample_freq is not None:
            df_recs["target_sample_freq"] = self.target_sample_freq
        df_recs["storage_path"] = df_recs["storage_path"].apply(
            lambda x: self.corpus_dir / x
        )
        recordings = RecordingSet(df_recs)

        logging.info("making ClassInfos")
        df_spks = df_meta[["speaker"]].drop_duplicates().sort_values(by="speaker")
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        speakers = ClassInfo(df_spks)

        df_langs = df_meta[["language"]].drop_duplicates().sort_values(by="language")
        df_langs.rename(columns={"language": "id"}, inplace=True)
        languages = ClassInfo(df_langs)

        df_codecs = df_meta[["codec"]].drop_duplicates().sort_values(by="codec")
        df_codecs.rename(columns={"codec": "id"}, inplace=True)
        codecs = ClassInfo(df_codecs)

        spoof_det = ClassInfo(pd.DataFrame({"id": ["bonafide", "spoof"]}))
        spoof_access = ClassInfo(pd.DataFrame({"id": ["LA", "PA"]}))

        classes = {
            "speaker": speakers,
            "language": languages,
            "spoof_det": spoof_det,
            "spoof_access": spoof_access,
            "codec": codecs,
        }

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes=classes,
            recordings=recordings,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers %f hours",
            len(segments),
            len(speakers),
            segments["duration"].sum() / 3600,
        )
