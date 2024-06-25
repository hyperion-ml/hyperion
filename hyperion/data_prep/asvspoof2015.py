"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep

spoof_system_dict = {
    "bonafide": "non spoof",
    "S1": "a naive frame-selection based voice conversion. It is an simplified version of the exemplar-based unit selection method proposed in [1].",
    "S2": "a simple voice conversion technique which only modifies the first coefficient of Mel-Cepstral coefficients.",
    "S3": "HMM-based speaker-adapted speech synthesis [2] implementated by HTS toolkit [3]. This system uses only 20 utterances to do adaptation.",
    "S4": "the same system as S3 but using 40 utterances from each target speaker to do adaptation.",
    "S5": "this is implemented by the voice conversion toolkit with in the publicly-available Festvox system [4]. It is based on a joint density Gaussian mixture model with maximum likelihood parameter generation considering global variance [5].",
    "S6": "a VC algorithm based on joint density Gaussian mixture models and maximum likelihood parameter generation considering global variance [26]",
    "S7": "a VC algorithm similar to S6, but using line spectrum pair (LSP) rather than mel-cepstral coeffecients for spectrum representation",
    "S8": "a tensor-based approach to VC [27] for which a Japanese dataset was used to construct the speaker space",
    "S9": "a VC algorithm which uses kernel-based partial least square (KPLS) to implement a non-linear transformation function [28] (without dynamic information, for simplification)",
    "S10": "an SS algorithm implemented with the open-source MARY Text-To-Speech system (MaryTTS)",
}

class ASVSpoof2015DataPrep(DataPrep):
    """Class for preparing ASVSpoof2015 database into tables,
       
    Attributes:
      corpus_dir: input data directory
      subset: train/dev/eval
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        self.subset = subset

    @staticmethod
    def dataset_name():
        return "asvspoof2015"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            default="train",
            choices=["train", "dev", "eval"],
            help="""if we prepare the data for [train, dev, eval]""",
        )

    def _get_metadata(self):
        if self.subset == "train":
            file_name = "cm_train.trn"
        elif self.subset == "dev":
            file_name = "cm_develop.ndx"
        elif self.subset == "eval":
            file_name = "cm_evaluation.ndx"

        file_path = self.corpus_dir / "CM_protocol" / file_name
        df_meta = pd.read_csv(file_path, sep=" ", header=None, names=["asvspoof_speaker", "file", "spoof_system", "spoof_det"])
        df_meta.loc[:,"asvspoof_speaker"] = df_meta["asvspoof_speaker"].apply(lambda x: f"ASVSpoof2015-{x}") 
        df_meta.loc[:,"spoof_det"] = df_meta["spoof_det"].apply(lambda x: "bonafide" if x=="human" else x)
        df_meta.loc[:,"spoof_system"] = df_meta["spoof_system"].apply(lambda x: "bonafide" if x=="human" else x)  
        def get_voco(x):
            if x == "bonafide":
                return "human"
            elif x == "S5":
                return "MLSA"
            elif x == "S10":
                return "waveform-concatenation"
            else:
                return "straight"
        df_meta.loc[:, "vocoder"] = df_meta["spoof_system"].apply(get_voco)
        #df_meta = df_meta[["id", "speaker", "spoof_det", "spoof_system"]]
        df_meta["language"] = "english"
        df_meta.set_index(df_meta.file, drop=False, inplace=True)
        return df_meta
        
    def prepare(self):
        logging.info(
            "Peparing ASVSpoof 2015 %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        logging.info("getting audio meta-data")
        df_meta = self._get_metadata()
        rec_dir = self.corpus_dir / "wav"
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.wav"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.wav", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"
        rec_files = [f for f in rec_files if f.with_suffix("").name in df_meta.index]
        assert len(rec_files) > 0, "recording files don't match metadata file"
        file_names = [f.with_suffix("").name for f in rec_files]
        if self.use_kaldi_ids:
            rec_ids = [
                    f'{df_meta.loc[f,"asvspoof_speaker"]}-{f}' for f in file_names
                ]
        else:
            rec_ids = file_names

        for id, file in zip(rec_ids, file_names):
            df_meta.loc[file,"id"] = id 

        # put id column first and remove file column
        cols = ["id"] + [c for c in df_meta.columns if c not in ["id", "file"]]
        df_meta = df_meta[cols]
        #df_meta.drop(columns=["file"], inplace=True)
            
        file_paths = [str(r) for r in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        recs = RecordingSet(recs)
        recs.sort()

        logging.info("getting recording durations")
        recs.get_durations(self.num_threads)
        #self.get_recording_duration(recs)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        segments = df_meta
        df_meta["duration"] = recs.loc[df_meta["id"], "duration"].values
    
        segments = SegmentSet(segments)
        segments.sort()

        logging.info("making speaker info file")
        uniq_speakers = np.unique(segments["asvspoof_speaker"])
        speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
            }
        )
        speakers = ClassInfo(speakers)

        logging.info("making spoofing/bonafide info file")
        spoof_det = ClassInfo(pd.DataFrame({"id": ["bonafide", "spoof"]}))

        logging.info("making spoofing tech info file")
        spoof_system = np.unique(segments["spoof_system"])
        spoof_system = ClassInfo(pd.DataFrame({"id": spoof_system, "description": [spoof_system_dict[i] for i in spoof_system]}))

        logging.info("making vocoder info file")
        vocoder = ClassInfo(pd.DataFrame({"id": segments["vocoder"].unique()}))

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes={"asvspoof_speaker": speakers, "spoof_det": spoof_det, "spoof_system": spoof_system, "vocoder": vocoder},
            recordings=recs,
            #enrollments=enrollments,
            #trials=trials,
            #sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )
