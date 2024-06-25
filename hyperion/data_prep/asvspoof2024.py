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
    "A01": ["TTS", "neural-waveform"],
    "A02": ["TTS", "vocoder"],
    "A03": ["TTS", "vocoder"],
    "A04": ["TTS", "waveform-concatenation"],
    "A05": ["VC", "vocoder"],
    "A06": ["VC", "spectral-filtering"],
    "A07": ["TTS", "vocoder+GAN"],
    "A08": ["TTS", "neural-waveform"],
    "A09": ["TTS", "vocoder"],
    "A10": ["TTS", "neural-waveform"],
    "A11": ["TTS", "griffin-lim"],
    "A12": ["TTS", "neural-waveform"],
    "A13": ["TTS_VC", "waveform-concatenation+waveform-filtering"],
    "A14": ["TTS_VC", "vocoder"],
    "A15": ["TTS_VC", "neural-waveform"],
    "A16": ["TTS", "waveform-concatenation"],
    "A17": ["VC", "waveform-filtering"],
    "A18": ["VC", "vocoder"],
    "A19": ["VC", "spectral-filtering"],
}


class ASVSpoof2024DataPrep(DataPrep):
    """Class for preparing ASVSpoof2021 database into tables,
       
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
            corpus_dir, output_dir, False, target_sample_freq, num_threads
        )

        self.subset = subset
        self.spoof_access = "LA"
       

    @staticmethod
    def dataset_name():
        return "asvspoof2024"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["train", "dev", "eval", "dev_enroll", "eval_enroll"],
            help="""if we prepare the data for ["train", "dev", "eval", "dev_enroll", "eval_enroll"]""",
            required=True,
        )


    def _get_metadata(self):

        file_path = self.corpus_dir / f"ASVspoof5.{self.subset}.metadata.txt"
        df_meta = pd.read_csv(file_path, sep=" ", header=None, names=["asvspoof_speaker", "file", "gender", "codec", "spoof_system", "spoof_det"], na_values="-")
        df_meta.loc[:,"asvspoof_speaker"] = df_meta["asvspoof_speaker"].apply(lambda x: f"ASVSpoof2024-{x}") 
        df_meta.loc[:, "spoof_access"] = df_meta["spoof_det"].apply(lambda x: None if x == "bonafide" else self.spoof_access)
        df_meta["language"] = "english"
        print(df_meta)
        
        def get_voco(x):
            if x in spoof_system_dict:
                return spoof_system_dict[x][1]
            else:
                return "human"
        df_meta.loc[:, "vocoder"] = df_meta["spoof_system"].apply(get_voco)
        def get_ttsvc(x):
            if x in spoof_system_dict:
                return spoof_system_dict[x][0]
            else:
                return None
        df_meta.loc[:, "spoof_method"] = df_meta["spoof_system"].apply(get_ttsvc)
        df_meta.set_index(df_meta.file, drop=False, inplace=True)
        return df_meta
    
    def make_trials(self):
        trials = {}
        dfs = []
        logging.info("making trials")
        file_path = self.corpus_dir / f"ASVspoof5.{self.subset}.trial.txt"
        columns = ["modelid", "segmentid", "targettype"]      
        df = pd.read_csv(
                file_path,
                header=None,
                sep=" ",
                names=columns,
            )
        spoof_idx = df["targettype"] == "spoof"
        df.loc[spoof_idx, "spoof_det"] = "spoof"
        df.loc[spoof_idx, "targettype"] = "nontarget"
            
        df.sort_values(by=["modelid", "segmentid"], inplace=True)
        file_path = self.output_dir / f"trials.csv"
        df.to_csv(file_path, index=False)
        dfs.append(df)
        trials["trials"] = file_path

        return trials
    
    def make_enrollments(self):
        logging.info("making enrollment map")
       
        if self.subset == "dev_enroll":
            file_path = self.corpus_dir / f"ASVspoof5.dev.enroll.txt"
        elif self.subset == "eval_enroll":
            file_path = self.corpus_dir / f"ASVspoof5.eval.enroll.txt"

        enrollments={}
        df_in = pd.read_csv(
                file_path,
                header=None,
                sep=" ",
                names=["modelid", "segmentid"],
        )
        model_ids = []
        segment_ids = []
        file_ids = []
        spks = []
        for _, row in df_in.iterrows():
            model = row["modelid"]
            segments = row["segmentid"].split(",")
            for segment in segments:
                model_ids.append(model)
                file_ids.append(segment)
                spk = f"ASVSpoof2024-{model}"
                    
                segment = f"ASVSpoof2024-{segment}"
                segment_ids.append(segment)
                spks.append(spk)

        df_out = pd.DataFrame({"modelid": model_ids, "segmentid": segment_ids, "file": file_ids, "asvspoof_speaker": spks})
        df_out.sort_values(by=["modelid", "segmentid"], inplace=True)
        segments = df_out.rename(columns={"segmentid": "id"})[["id", "asvspoof_speaker", "file"]]
        file_path = self.output_dir / "enroll.csv"
        df_out.drop(columns=["file", "asvspoof_speaker"], inplace=True)
        df_out.to_csv(file_path, index=False)
        enrollments["enroll"] = file_path
        segments["spoof_det"] = "bonafide"
        return segments, enrollments
    

    def prepare(self):
        logging.info(
            "Peparing ASVSpoof 2024 %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        if "enroll" in self.subset:
            self.prepare_enroll()
        else:
            self.prepare_test()

    def prepare_enroll(self):
        logging.info("getting enrollment data")
        df_meta, enrollments = self.make_enrollments()

        if self.subset == "dev_enroll":
            rec_dir = self.corpus_dir / "flac_D"

        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.flac"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.flac", recursive=True)
            ]

        df_meta.set_index(df_meta.file, inplace=True)
        assert len(rec_files) > 0, "recording files not found"
        rec_files = [f for f in rec_files if f.with_suffix("").name in df_meta.index]
        assert len(rec_files) > 0, "recording files don't match metadata file"
        file_names = [f.with_suffix("").name for f in rec_files]

        rec_ids = [
                    f"ASVSpoof2024-{f}" for f in file_names
                ]

        df_meta.drop(columns=["file"], inplace=True)
            
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

        df_meta["duration"] = recs.loc[df_meta["id"], "duration"].values
        logging.info("making SegmentsSet")
        segments = df_meta
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

        classes = {"asvspoof_speaker": speakers, "spoof_det": spoof_det}
        
        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes=classes,
            recordings=recs,
            enrollments=enrollments,
            trials=None,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )

    def prepare_test(self):
        logging.info("getting audio meta-data")
        df_meta = self._get_metadata()

        if self.subset == "train":
            rec_dir = self.corpus_dir / f"flac_T"
        elif self.subset == "dev":
            rec_dir = self.corpus_dir / f"flac_D"

        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.flac"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.flac", recursive=True)
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

        spk_map = df_meta[["file", "asvspoof_speaker"]]
        spk_map.set_index(spk_map.file, inplace=True)
        print(df_meta, spk_map)

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

        df_meta["duration"] = recs.loc[df_meta["id"], "duration"].values
        logging.info("making SegmentsSet")
        segments = df_meta
        segments = SegmentSet(segments)
        segments.sort()

        print(segments)

        # logging.info("making speaker info file")
        # uniq_speakers = np.unique(segments["speaker"])
        # speakers = pd.DataFrame(
        #     {
        #         "id": uniq_speakers,
        #     }
        # )
        # speakers = ClassInfo(speakers)

        uniq_speakers = np.unique(segments["asvspoof_speaker"])
        asvspoof_speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
            }
        )
        asvspoof_speakers = ClassInfo(asvspoof_speakers)

        logging.info("making spoofing/bonafide info file")
        spoof_det = ClassInfo(pd.DataFrame({"id": ["bonafide", "spoof"]}))

        logging.info("making vocoder info file")
        vocoder = ClassInfo(pd.DataFrame({"id": segments["vocoder"].unique()}))

        logging.info("making spoof_system info file")
        spoof_system = ClassInfo(pd.DataFrame({"id": segments["spoof_system"].unique()}))

        logging.info("making spoof_method info file")
        spoof_method = ClassInfo(pd.DataFrame({"id": segments["spoof_method"].unique()}))

        logging.info("making spoof access info file")
        spoof_access = ClassInfo(pd.DataFrame({"id": ["LA", "PA"]}))
             
        classes = {"asvspoof_speaker": asvspoof_speakers, "spoof_det": spoof_det, "spoof_access": spoof_access, "spoof_system": spoof_system,"spoof_method": spoof_method, "vocoder": vocoder}
        # if self.spoof_access == "PA":
        #     logging.info("making environment info file")
        #     environment = np.unique(segments["environment"].dropna())
        #     environment = ClassInfo(pd.DataFrame({"id": environment}))
        #     classes["environment"] = environment

        if self.subset in ["dev", "eval"]:
            trials = self.make_trials()
        else:
            trials = None
       
        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes=classes,
            recordings=recs,
            enrollments=None,
            trials=trials,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(asvspoof_speakers)
        )
