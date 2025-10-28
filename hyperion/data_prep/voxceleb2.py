"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo, ArgumentParser
from tqdm import tqdm

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class VoxCeleb2DataPrep(DataPrep):
    """
    Class for preparing the VoxCeleb2 dataset into structured tables.

    Attributes:
        corpus_dir (PathLike): Input data directory.
        subset (str): 'dev' or 'test' split of the dataset.
        cat_videos (bool): Concatenate utterances from the same video.
        enrichment_metadata (bool): Whether to enrich with external age/gender metadata.
        output_dir (PathLike): Output directory for processed data.
        use_kaldi_ids (bool): Use Kaldi-style segment IDs (speaker-segment).
        target_sample_freq (Optional[int]): Target sample rate in Hz.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        cat_videos: bool = False,
        enrichment_metadata: bool = False,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ) -> None:
        """
        Initialize the VoxCeleb2DataPrep instance.
        """
        use_kaldi_ids = True
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        self.subset = subset
        self.cat_videos = cat_videos
        self.enrichment_metadata = enrichment_metadata

    @staticmethod
    def dataset_name() -> str:
        """Returns the name of the dataset."""
        return "voxceleb2"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """
        Adds dataset-specific arguments to the argument parser.
        """
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
        parser.add_argument(
            "--enrichment-metadata",
            default=False,
            action=ActionYesNo,
            help="""enrich metadata with VoxCeleb2 age and gender estimation""",
        )

    def _get_metadata(self) -> pd.DataFrame:
        """Downloads and returns VoxCeleb2 metadata as a DataFrame."""
        file_name = "vox2_meta.csv"
        file_path = self.corpus_dir / file_name
        if not file_path.exists():
            file_path = self.output_dir / file_name
            if not file_path.exists():
                url = "https://www.openslr.org/resources/49/vox2_meta.csv"
                file_path, _ = urlretrieve_progress(url, file_path, desc=file_name)

        df_meta = pd.read_csv(file_path, sep="\t")
        df_meta.rename(columns=str.strip, inplace=True)
        df_meta = df_meta.applymap(lambda x: str.strip(x) if isinstance(x, str) else x)
        df_meta.set_index("VoxCeleb2 ID", inplace=True)
        return df_meta

    def _get_enrichment_metadata(
        self, segments: SegmentSet, speakers: ClassInfo
    ) -> None:
        """
        Adds enriched metadata (e.g., age, nationality) to segments and speakers.

        Args:
            segments (SegmentSet): Segment-level metadata.
            speakers (ClassInfo): Speaker-level metadata.
        """
        import subprocess

        repo_url = (
            "https://github.com/jesus-villalba/voxceleb_enrichment_age_gender.git"
        )
        target_dir = self.output_dir / "voxceleb_enrichment_age_gender"

        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)

        ext_table = pd.read_csv(
            "voxceleb_enrichment_age_gender/dataset/final_dataframe_extended.csv",
            sep=",",
        )
        """Ext table columns:
        Name,gender_wiki,birth_date_wiki,nationality_wiki,
        gender_dbpedia,birth_date_dbpedia,nationality_dbpedia,
        gender_gkg,birth_date_gkg,nationality_gkg,
        video_id,title,publishing_date,description,year_in_title,
        VoxCeleb_ID,gender,birth_year,year_upload_yt,
        recording_year,recording_year_title_only,
        speaker_age,speaker_age_title_only
        """
        ext_table.rename(columns={"nationality_wiki": "nationality"}, inplace=True)
        columns = [
            "video_id",
            "nationality",
            "birth_year",
            "recording_year",
            "recording_year_title_only",
            "speaker_age",
            "speaker_age_title_only",
            "VoxCeleb_ID",
        ]

        segments.add_columns(
            right_table=ext_table[columns],
            on=["speaker", "video_id"],
            right_on=["VoxCeleb_ID", "video_id"],
        )
        segments.loc[
            (segments["speaker_age"] >= 18) & (segments["speaker_age"] <= 24),
            "arts_age_range",
        ] = "young"
        segments.loc[
            (segments["speaker_age"] >= 35) & (segments["speaker_age"] <= 44),
            "arts_age_range",
        ] = "adult"
        segments.loc[
            (segments["speaker_age"] >= 55) & (segments["speaker_age"] <= 64),
            "arts_age_range",
        ] = "senior"
        segments.loc[segments["speaker_age"] <= 24, "arts_ext_age_range"] = "young"
        segments.loc[
            (segments["speaker_age"] >= 35) & (segments["speaker_age"] <= 44),
            "arts_ext_age_range",
        ] = "adult"
        segments.loc[segments["speaker_age"] >= 55, "arts_ext_age_range"] = "senior"

        segments.loc[
            (segments["speaker_age_title_only"] >= 18)
            & (segments["speaker_age_title_only"] <= 24),
            "arts_age_range_title_only",
        ] = "young"
        segments.loc[
            (segments["speaker_age_title_only"] >= 35)
            & (segments["speaker_age_title_only"] <= 44),
            "arts_age_range_title_only",
        ] = "adult"
        segments.loc[
            (segments["speaker_age_title_only"] >= 55)
            & (segments["speaker_age_title_only"] <= 64),
            "arts_age_range_title_only",
        ] = "senior"
        segments.loc[
            segments["speaker_age_title_only"] <= 24, "arts_ext_age_range_title_only"
        ] = "young"
        segments.loc[
            (segments["speaker_age_title_only"] >= 35)
            & (segments["speaker_age_title_only"] <= 44),
            "arts_ext_age_range_title_only",
        ] = "adult"
        segments.loc[
            segments["speaker_age_title_only"] >= 55, "arts_ext_age_range_title_only"
        ] = "senior"

        speakers_ext = segments.df[
            ["speaker", "nationality", "birth_year"]
        ].drop_duplicates()
        speakers.add_columns(right_table=speakers_ext, on=["id"], right_on=["speaker"])

    def _get_langs_est(self) -> pd.DataFrame:
        """Downloads and returns language estimations for each segment."""
        file_name = "lang_vox2_final.csv"
        file_path = self.corpus_dir / file_name
        if not file_path.exists():
            file_path = self.output_dir / file_name
            if not file_path.exists():
                url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/lang_vox2_final.csv"
                file_path, _ = urlretrieve_progress(url, file_path, desc=file_name)

        df_lang = pd.read_csv(file_path, sep=",")

        if self.cat_videos:

            def get_video(x):
                x = re.sub("/[^/]*.wav$", "", x)
                return re.sub("/", "-", x)

        elif self.use_kaldi_ids:

            def get_video(x):
                x = re.sub(".wav$", "", x)
                return re.sub("/", "-", x)

        else:

            def get_video(x):
                x = re.sub(".wav$", "", x)
                x = re.sub("^[^/]*/", "", x)
                return re.sub("/", "-", x)

        df_lang["id"] = df_lang["filename"].apply(get_video)
        df_lang.drop(["filename"], axis=1, inplace=True)
        df_lang.drop_duplicates(inplace=True)
        df_lang.set_index("id", inplace=True)
        df_lang["lang"] = df_lang["lang"].apply(str.lower)
        return df_lang

    @staticmethod
    def make_cat_list(
        lists_cat_dir: Path,
        rec_id: str,
        rec_files: List[Path],
        video_idx: np.ndarray,
        i: int,
    ) -> str:
        """
        Create a text list for ffmpeg to concatenate multiple recordings into one.

        Args:
            lists_cat_dir (Path): Directory to write the concat file.
            rec_id (str): The resulting recording ID.
            rec_files (List[Path]): List of all recording paths.
            video_idx (np.ndarray): Array mapping each file to a video index.
            i (int): Current video index to process.

        Returns:
            str: ffmpeg pipe command.
        """
        list_file = lists_cat_dir / f"{rec_id}.txt"
        with open(list_file, "w") as fw:
            rec_idx = (video_idx == i).nonzero()[0]
            recs_i = [f"file {rec_files[j]}" for j in rec_idx]
            recs_i.sort()
            recs_i = "\n".join(recs_i)
            fw.write(f"{recs_i}\n")

        file_path = (
            f"ffmpeg -v 8 -f concat -safe 0 -i {list_file} -f wav -acodec pcm_s16le -|"
        )
        return file_path

    def prepare(self) -> None:
        """
        Runs the full preparation pipeline:
        - Scans audio files
        - Assigns IDs
        - Computes durations
        - Creates RecordingSet and SegmentSet
        - Optionally adds enrichment metadata
        - Saves HypDataset
        """
        logging.info(
            "Peparing VoxCeleb2 %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )
        logging.info("getting audio meta-data")
        df_meta = self._get_metadata()
        logging.info("getting language estimations")
        df_lang = self._get_langs_est()
        rec_dir = self.corpus_dir / self.subset
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = [Path(f) for f in glob.iglob(f"{rec_dir}/**/*.m4a", recursive=True)]
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.m4a", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"

        speakers = [f.parents[1].name for f in rec_files]
        video_ids = [f.parent.name for f in rec_files]
        if self.cat_videos:
            rec_ids = [f"{s}-{v}" for s, v in zip(speakers, video_ids)]
            lists_cat_dir = self.output_dir / "lists_cat"
            lists_cat_dir.mkdir(exist_ok=True, parents=True)
            rec_ids, uniq_rec_idx, rec_idx = np.unique(
                rec_ids, return_index=True, return_inverse=True
            )
            speakers = [speakers[i] for i in uniq_rec_idx]
            video_ids = [video_ids[i] for i in uniq_rec_idx]

            file_paths = []
            futures = []
            logging.info("making video cat lists")
            logging.info("submitting threats...")
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                for i, rec_id in tqdm(enumerate(rec_ids)):
                    future = pool.submit(
                        VoxCeleb2DataPrep.make_cat_list,
                        lists_cat_dir,
                        rec_id,
                        rec_files,
                        rec_idx,
                        i,
                    )
                    futures.append(future)

            logging.info("waiting threats...")
            file_paths = [f.result() for f in tqdm(futures)]
        else:
            file_names = [f.with_suffix("").name for f in rec_files]
            if self.use_kaldi_ids:
                rec_ids = [
                    f"{s}-{v}-{f}" for s, v, f in zip(speakers, video_ids, file_names)
                ]
            else:
                rec_ids = [f"{v}-{f}" for v, f in zip(video_ids, file_names)]

            file_paths = []
            logging.info("making pipe commands")
            for rec_file in tqdm(rec_files):
                file_path = f"ffmpeg -v 8 -i {rec_file} -f wav -acodec pcm_s16le - |"
                file_paths.append(file_path)

        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        recs = RecordingSet(recs)
        recs.sort()

        logging.info("getting recording durations")
        self.get_recording_duration(recs)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        segments = pd.DataFrame(
            {
                "id": rec_ids,
                "video_id": video_ids,
                "speaker": speakers,
                "gender": df_meta.loc[speakers, "Gender"],
                "language_est": [
                    df_lang.loc[r, "lang"] if r in df_lang.index else "N/A"
                    for r in rec_ids
                ],
                "language_est_conf": [
                    df_lang.loc[r, "confidence"] if r in df_lang.index else "N/A"
                    for r in rec_ids
                ],
                "duration": recs.loc[rec_ids, "duration"].values,
                "corpusid": "voxceleb",
                "dataset": self.dataset_name(),
                "source_type": "afv",
                "original_bandwidth": 8000,
            }
        )
        segments = SegmentSet(segments)
        segments.sort()

        logging.info("making speaker info file")
        uniq_speakers = np.unique(speakers)
        speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
                "vgg_id": df_meta.loc[uniq_speakers, "VGGFace2 ID"],
                "gender": df_meta.loc[uniq_speakers, "Gender"],
            }
        )
        speakers = ClassInfo(speakers)

        logging.info("making language info file")
        languages = np.unique(df_lang["lang"])
        languages = ClassInfo(pd.DataFrame({"id": languages}))

        logging.info("making gender info file")
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        if self.enrichment_metadata:
            self._get_enrichment_metadata(segments, speakers)

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            {"speaker": speakers, "language_est": languages, "gender": genders},
            recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )
