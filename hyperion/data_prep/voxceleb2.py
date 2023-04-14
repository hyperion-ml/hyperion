"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ActionYesNo
from pathlib import Path
import re

import pandas as pd
import numpy as np

from ..utils.misc import urlretrieve_progress
from ..utils import RecordingSet, SegmentSet, ClassInfo
from .data_prep import DataPrep


class VoxCeleb2DataPrep(DataPrep):
    """Class for preparing VoxCeleb2 database into tables

    Attributes:
      corpus_dir: input data directory
      subset: subset of the data dev or test
      cat_videos: concatenate utterances from the same video.
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir,
        subset,
        cat_videos,
        output_dir,
        use_kaldi_ids,
        target_sample_freq,
    ):
        super().__init__(corpus_dir, output_dir, use_kaldi_ids, target_sample_freq)
        self.subset = subset
        self.cat_videos = cat_videos

    @staticmethod
    def dataset_name():
        return "voxceleb2"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            default="dev",
            choices=["dev", "test"],
            help="""vox2 subset in [dev, test]""",
        )
        parser.add_argument(
            "--cat-videos",
            default=False,
            action=ActionYesNo,
            help="""concatenate utterances from the same video.""",
        )

    def _get_metadata(self):
        file_name = "vox2_meta.csv"
        file_path = self.corpus_dir / file_name
        if not file_path.exists():
            file_path = self.output_dir / file_name
            if not file_path.exists():
                url = "https://www.openslr.org/resources/49/vox2_meta.csv"
                file_path, _ = urlretrieve_progress(url, file_path, desc=file_name)

        df_meta = pd.read_csv(file_path, sep="\t")
        print(df_meta.head())
        df_meta.set_index("VoxCeleb2 ID")
        return df_meta

    def _get_langs_est(self):
        file_name = "lang_vox2_final.csv"
        file_path = self.corpus_dir / file_name
        if not file_path.exists():
            file_path = self.output_dir / file_name
            if not file_path.exists():
                url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/lang_vox2_final.csv"
                file_path, _ = urlretrieve_progress(url, file_path, desc=file_name)

        df_lang = pd.read_csv(file_path, sep=",")

        def get_video(x):
            x = re.sub("/.*.wav$", "", x)
            x = re.sub("^.*/", "", x)
            return x

        df_lang["video"] = df_lang["filename"].apply(get_video)
        df_lang["filename"].drop(["filename"], axis=1, inplace=True)
        df_lang.drop_duplicates(inplace=True)
        df_lang.set_index("video")
        return df_lang

    def prepare(self):
        df_meta = self._get_metadata()
        df_lang = self._get_langs_est()
        rec_dir = self.corpus_dir / self.subset
        rec_files = list(rec_dir.glob("**/*.m4a"))
        speakers = [f.parents[1].name for f in rec_files]
        video_ids = [f.parent.name for f in rec_files]
        if self.concat_videos:
            lists_cat_dir = self.output_dir / "lists_cat"
            lists_cat_dir.mkdir(exist_ok=True, parents=True)
            uniq_video_ids, uniq_video_idx, video_idx = np.unique(
                video_ids, return_index=True, return_inverse=True
            )
            rec_ids = uniq_video_ids
            speakers = speakers[uniq_video_idx]
            if self.use_kaldi_ids:
                rec_ids = [f"{s}-{v}" for s, v in zip(speakers, uniq_video_ids)]
            else:
                rec_ids = uniq_video_ids

            file_paths = []
            for i, video_id in enumerate(uniq_video_ids):
                list_file = lists_cat_dir / f"{video_id}.txt"
                with open(list_file, "w") as fw:
                    rec_mask = video_idx == i
                    recs_i = rec_files[rec_mask]
                    for rec in recs_i:
                        fw.write(f"{rec}\n")

                file_path = f"ffmpeg -v 8 -f concat -safe 0 -i {list_file} -f wav -acodec pcm_s16le -|"
                file_paths.append(file_path)

            video_ids = uniq_video_ids

        else:
            file_names = [f.name for f in rec_files]
            if self.use_kaldi_ids:
                rec_ids = [
                    f"{s}-{v}-{f}" for s, v, f in zip(speakers, video_ids, file_names)
                ]
            else:
                rec_ids = [f"{v}-{f}" for v, f in zip(video_ids, file_names)]

            file_paths = []
            for rec_file in rec_files:
                file_path = f"ffmpeg -v 8 -i {rec_file} -f wav -acodec pcm_s16le - |"
                file_paths.append(file_path)

        recs = pd.DataFrame({"id": rec_ids, "file_path": file_paths})
        recs = RecordingSet(recs)
        segments = pd.DataFrame(
            {
                "id": rec_ids,
                "video_ids": video_ids,
                "speaker": speakers,
                "gender": df_meta.loc[speakers, "Gender"],
            }
        )
        segments = SegmentSet(segments)
        uniq_speakers = np.unique(speakers)
        speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
                "vgg_id": df_meta.loc[uniq_speakers, "VGGFace2 ID"],
                "gender": df_meta.loc[uniq_speakers, "Gender"],
            }
        )
        speakers = ClassInfo(speakers)

        print(recs)
        print(segments)
        print(speakers)
