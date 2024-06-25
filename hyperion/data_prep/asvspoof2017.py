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

sentence_dict = {
    "S01": 'My voice is my password',
    "S02": 'OK Google',
    "S03": 'Only lawyers love millionaires',
    "S04": 'Artificial intelligence is for real',
    "S05": 'Birthday parties have cupcakes and ice cream',
    "S06": 'Actions speak louder than words',
    "S07": 'There is no such thing as a free lunch',
    "S08": 'A watched pot never boils',
    "S09": 'Jealousy has twenty-twenty vision',
    "S10": 'Necessity is the mother of invention',
}

env_dict = {
    "E01": 'Anechoic room',
    "E02": 'Balcony 01',
    "E03": 'Balcony 02',
    "E04": 'Home 07',
    "E05": 'Home 08',
    "E06": 'Cantine',
    "E07": 'Home 01',
    "E08": 'Home 02',
    "E09": 'Home 03',
    "E10": 'Home 04',
    "E11": 'Home 05',
    "E12": 'Home 06',
    "E13": 'Office 01',
    "E14": 'Office 02',
    "E15": 'Office 03',
    "E16": 'Office 04',
    "E17": 'Office 05',
    "E18": 'Office 06',
    "E19": 'Office 07',
    "E20": 'Office 08',
    "E21": 'Office 09',
    "E22": 'Office 10',
    "E23": 'Studio',
    "E24": 'Analog wire 01',
    "E25": 'Analog wire 02',
    "E26": 'Analog wire 03',
}

playback_device_dict = {
    "P01": 'All-in-one PC speakers',
    "P02": 'Creative A60 speakers',
    "P03": 'Genelec 8020C studio monitor',
    "P04": 'Genelec 8020C studio monitor (2 speakers)',
    "P05": 'Beyerdynamic DT 770 PRO headphones',
    "P06": 'Dell laptop internal speakers',
    "P07": 'Dynaudio BM5A speaker',
    "P08": 'HP Laptop internal speakers',
    "P09": 'VIFA M10MD-39-08 speaker',
    "P10": 'ACER netbook internal speakers',
    "P11": 'BQ Aquaris M5 smartphone',
    "P12": 'Logitech low quality speakers',
    "P13": 'Desktop PC line output',
    "P14": 'Labtec LCS-1050 speakers',
    "P15": 'Edirol MA-15D studio monitor',
    "P16": 'Lenovo Ideatab S6000-H tablet',
    "P17": 'Logitech S120 multimedia speakers',
    "P18": 'MacBook pro internal speakers',
    "P19": 'Altec lansing Orbit USB iML227 portable speaker',
    "P20": 'Samsung GT-I9100 smartphone',
    "P21": 'Samsung GT-P6200 tablet',
    "P22": 'Behringer Truth B2030A studio monitor',
    "P23": 'Focusrite Scarlett 2i2 audio interface line output',
    "P24": 'Focusrite Scarlett 2i4 audio interface line output',
    "P25": 'Genelec 6010A studio monitor',
    "P26": 'AKG K242HD Headset',

}

recording_device_dict = {
    "R01": 'Zoom H6 handy recorder',
    "R02": 'BQ Aquaris M5 smartphone',
    "R03": 'Low-quality headset',
    "R04": 'Nokia Lumia 635 smartphone',
    "R05": 'Røde NT2 microphone',
    "R06": 'Røde smartLav+ microphone',
    "R07": 'Samsung Galaxy 7s smartphone',
    "R08": 'Desktop PC microphone input',
    "R09": 'Zoom H6 handy recorder with Behringer ECM8000 microphone',
    "R10": 'Zoom H6 handy recorder with MSH-6 microphone',
    "R11": 'Zoom H6 handy recorder with XY microphone',
    "R12": 'iPhone 5c smartphone',
    "R13": 'iPhone 7 plus smartphone',
    "R14": 'iPhone 4 smartphone',
    "R15": 'Logitech C920 webcam',
    "R16": 'miniDSP UMIK-1 microphone',
    "R17": 'Samsung Galaxy Trend 2 smartphone',
    "R18": 'Samsung GT-I9100 smartphone',
    "R19": 'Samsung GT-P6200 tablet',
    "R20": 'Samsung Trend 2 smartphone',
    "R21": 'AKG C3000 microphone',
    "R22": 'SE electronic 2200a microphone',
    "R23": 'Focusrite Scarlett 2i2 interface line input',
    "R24": 'Focusrite Scarlett 2i4 interface line input',
    "R25": 'Zoom HD1 handy recorder',
}

class ASVSpoof2017DataPrep(DataPrep):
    """Class for preparing ASVSpoof2017 database into tables,
       
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
        return "asvspoof2017"

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
            file_name = "ASVspoof2017_V2_train.trn.txt"
        elif self.subset == "dev":
            file_name = "ASVspoof2017_V2_dev.trl.txt"
        elif self.subset == "eval":
            file_name = "ASVspoof2017_V2_eval.trl.txt"
    
        file_path = self.corpus_dir / "protocol_V2" / file_name
        df_meta = pd.read_csv(file_path, sep=" ", header=None, names=["file", "spoof_det", "speaker", "reddots_sentence", "environment", "playback_device", "recording_device"], na_values="-")
        df_meta.loc[:,"speaker"] = df_meta["speaker"].apply(lambda x: f"RedDots-{x}") 
        df_meta.loc[:,"spoof_det"] = df_meta["spoof_det"].apply(lambda x: "bonafide" if x=="genuine" else x)
        df_meta.loc[:,"text"] = df_meta["reddots_sentence"].apply(lambda x: sentence_dict[x])
        df_meta["language"] = "english"
        df_meta["vocoder"] = "human"
        df_meta.set_index(df_meta.file, drop=False, inplace=True)
        return df_meta
        
    def prepare(self):
        logging.info(
            "Peparing ASVSpoof 2017 %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        logging.info("getting audio meta-data")
        df_meta = self._get_metadata()
        rec_dir = self.corpus_dir / f"ASVspoof2017_V2_{self.subset}"
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.wav"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.wav", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"
        rec_files = [f for f in rec_files if f.name in df_meta.index]
        assert len(rec_files) > 0, "recording files don't match metadata file"
        file_names = [f.name for f in rec_files]
        print(df_meta)
        #print(file_names)
        if self.use_kaldi_ids:
            rec_ids = [
                    f'{df_meta.loc[f,"speaker"]}-{f}' for f in file_names
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
        uniq_speakers = np.unique(segments["speaker"])
        speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
            }
        )
        speakers = ClassInfo(speakers)

        logging.info("making spoofing/bonafide info file")
        spoof_det = ClassInfo(pd.DataFrame({"id": ["bonafide", "spoof"]}))

        logging.info("making reddots sentence file")
        sentence = np.unique(segments["reddots_sentence"])
        sentence = ClassInfo(pd.DataFrame({"id": sentence, "text": [sentence_dict[i] for i in sentence]}))

        logging.info("making environment info file")
        environment = np.unique(segments["environment"].dropna())
        environment = ClassInfo(pd.DataFrame({"id": environment, "description": [env_dict[i] for i in environment]}))

        logging.info("making playback device info file")
        playback_dev = np.unique(segments["playback_device"].dropna())
        playback_dev = ClassInfo(pd.DataFrame({"id": playback_dev, "description": [playback_device_dict[i] for i in playback_dev]}))

        logging.info("making recording device info file")
        recording_dev = np.unique(segments["recording_device"].dropna())
        recording_dev = ClassInfo(pd.DataFrame({"id": recording_dev, "description": [recording_device_dict[i] for i in recording_dev]}))

        logging.info("making vocoder info file")
        vocoder = ClassInfo(pd.DataFrame({"id": ["human"]}))

        # if self.task == "test":
        #     enrollments, trials = self.make_trials()

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes={"speaker": speakers, "spoof_det": spoof_det, "reddots_sentence": sentence, "environment": environment, "playback_device": playback_dev, "recording_device": recording_dev, "vocoder": vocoder},
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
