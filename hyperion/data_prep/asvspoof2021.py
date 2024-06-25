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


class ASVSpoof2021DataPrep(DataPrep):
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
        if "la" in self.subset:
            self.spoof_access = "LA"  
        elif "pa" in self.subset:
            self.spoof_access = "PA"
        else:
            self.spoof_access = "DF"

        self.subsubset = self.subset[3:]

    @staticmethod
    def dataset_name():
        return "asvspoof2021"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["la_eval", "pa_eval", "df_eval"],
            help="""if we prepare the data for ["la_eval", "pa_eval", "df_eval"]""",
            required=True,
        )

    def _get_vctk_vcc_metadata(self, df_meta):
        file_path = self.corpus_dir / f"ASVspoof2021_{self.spoof_access}_VCTK_MetaInfo.tsv"
        df_meta2 = pd.read_csv(file_path, sep="\t")
        if self.spoof_access == "DF":
            file_path = self.corpus_dir / f"ASVspoof2021_{self.spoof_access}_VCC_MetaInfo.tsv"
            df_meta3 = pd.read_csv(file_path, sep="\t")
            df_meta2 = pd.concat((df_meta2, df_meta3), ignore_index=True)

        df_meta = pd.merge(df_meta, df_meta2, how="left", left_index=True, right_on="ASVspoof_ID")
        df_meta.set_index(df_meta.file, inplace=True)
        rename_cols = {c: c.lower() for c in df_meta2.columns}
        df_meta.rename(columns=rename_cols, inplace=True)
        if self.spoof_access in ["LA", "DF"]:
            df_meta.rename(columns={"tts_text": "text"}, inplace=True)
            idx = df_meta["spoof_det"] == "spoof"
            df_meta.loc[idx, "speaker"] = df_meta.loc[idx,"tts_vc_target_speaker"].apply(lambda x: f"VCTK-{x}")
            idx = df_meta["spoof_det"] == "bonafide"
            df_meta.loc[idx, "speaker"] = df_meta.loc[idx, "vctk_id"].apply(lambda x: f"VCTK-{x[:4]}" if isinstance(x, str) else x)
            # don't know how to get speakers for VCC
        else:
            df_meta["speaker"] = df_meta["vctk_id"].apply(lambda x: f"VCTK-{x[:4]}" if isinstance(x, str) else x)

        na_idx = df_meta["speaker"].isnull()
        df_meta.loc[na_idx, "speaker"] = df_meta.loc[na_idx, "asvspoof_speaker"]

        return df_meta

    def _get_metadata(self):
        
        file_path = self.corpus_dir / "keys"/ self.spoof_access / "CM" / f"trial_metadata.txt"
        if self.spoof_access == "LA":
            df_meta = pd.read_csv(file_path, sep=" ", header=None, names=["asvspoof_speaker", "file", "codec", "tx_location", "spoof_system", "spoof_det", "trim", "subset"], na_values="-")
        elif self.spoof_access == "PA":
            df_meta = pd.read_csv(file_path, sep=" ", header=None, names=["asvspoof_speaker", "file", "asv_room", "asv_mic", "asv_distance", "adq_room", "adq_mic", "replay_device", "adq_distance", "spoof_det", "trim", "subset"], na_values="-")
        elif self.spoof_access == "DF":
            df_meta = pd.read_csv(file_path, sep=" ", header=None, names=["asvspoof_speaker", "file", "codec", "source_dataset", "spoof_system", "spoof_det", "trim", "subset", "vocoder", "u1", "u2", "u3", "u4"], na_values="-")
            df_meta.drop(columns=["u1", "u2", "u3", "u4"], inplace=True)

        df_meta.loc[:,"asvspoof_speaker"] = df_meta["asvspoof_speaker"].apply(lambda x: f"ASVSpoof2021-{x}") 
        df_meta.loc[:, "spoof_access"] = df_meta["spoof_det"].apply(lambda x: None if x == "bonafide" else self.spoof_access)
        df_meta["language"] = "english"
        print(df_meta)
        if self.spoof_access != "DF":
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
        df_meta = self._get_vctk_vcc_metadata(df_meta)
        return df_meta
    
    def make_trials(self):
        # def get_kaldi_segmentid(s):
        #     return f'{spk_map.loc[s, "asvspoof_speaker"]}-{s}'
        
        trials = {}
        dfs = []
        logging.info("making trials")
        file_path = self.corpus_dir / "keys" / self.spoof_access / "ASV" / "trial_metadata.txt"
        if self.spoof_access == "LA":
            columns = ["modelid", "segmentid", "codec", "tx_location", "spoof_system", "targettype", "trim", "subset"]
        else:
            columns = ["modelid", "segmentid", "asv_room", "asv_mic", "asv_distance", "adq_room", "adq_mic", "replay_device", "adq_distance", "targettype", "trim", "subset"]
            
        df = pd.read_csv(
                file_path,
                header=None,
                sep=" ",
                names=columns,
            )
        spoof_idx = df["targettype"] == "spoof"
        df.loc[spoof_idx, "spoof_det"] = "spoof"
        df.loc[spoof_idx, "targettype"] = "nontarget"
            
        # if self.use_kaldi_ids:
            #df["segmentid"] = df["segmentid"].apply(get_kaldi_segmentid)

        # if self.spoof_access == "LA":
        #     df.drop(columns=["spoof_det"], inplace=True)
        # else:
        #     df.drop(columns=["environment", "spoof_det"], inplace=True)
            
        df.sort_values(by=["modelid", "segmentid"], inplace=True)
        file_path = self.output_dir / f"trials.csv"
        df.to_csv(file_path, index=False)
        dfs.append(df)
        trials["trials"] = file_path
        return trials
    

    def prepare(self):
        logging.info(
            "Peparing ASVSpoof 2021 %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )
        logging.info("getting audio meta-data")
        df_meta = self._get_metadata()

        rec_dir = self.corpus_dir / f"ASVspoof2021_{self.spoof_access}_{self.subsubset}"
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

        logging.info("making speaker info file")
        uniq_speakers = np.unique(segments["speaker"])
        speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
            }
        )
        speakers = ClassInfo(speakers)

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
             
        classes = {"speaker": speakers, "asvspoof_speaker": asvspoof_speakers, "spoof_det": spoof_det, "spoof_access": spoof_access, "spoof_system": spoof_system,"spoof_method": spoof_method, "vocoder": vocoder}
        # if self.spoof_access == "PA":
        #     logging.info("making environment info file")
        #     environment = np.unique(segments["environment"].dropna())
        #     environment = ClassInfo(pd.DataFrame({"id": environment}))
        #     classes["environment"] = environment

        if self.subset in ["la_eval", "pa_eval"]:
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
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )
