"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import glob

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import ClassInfo, Dataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class VoxCeleb1DataPrep(DataPrep):
    """Class for preparing VoxCeleb1 database into tables,
       It prepares the full voxceleb either to train or test with
       Original/Entire/Hard.
       We don't consider preparing dev for train and test for test Original

    Attributes:
      corpus_dir: input data directory
      task: train/test
      cat_videos: concatenate utterances from the same video.
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        task: str,
        cat_videos: bool,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
    ):
        use_kaldi_ids = True
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        self.task = task
        assert (
            cat_videos == False or task == "train"
        ), "cat-videos is only available for train task"

        self.cat_videos = cat_videos

    @staticmethod
    def dataset_name():
        return "voxceleb1"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--task",
            default="test",
            choices=["test", "train"],
            help="""if we prepare the data for [test, train]""",
        )
        parser.add_argument(
            "--cat-videos",
            default=False,
            action=ActionYesNo,
            help="""concatenate utterances from the same video.""",
        )

    def _get_metadata(self):
        file_name = "vox1_meta.csv"
        file_path = self.corpus_dir / file_name
        if not file_path.exists():
            file_path = self.output_dir / file_name
            if not file_path.exists():
                url = "https://www.openslr.org/resources/49/vox1_meta.csv"
                file_path, _ = urlretrieve_progress(url, file_path, desc=file_name)

        df_meta = pd.read_csv(file_path, sep="\t")
        df_meta.rename(columns=str.strip, inplace=True)
        df_meta = df_meta.applymap(lambda x: str.strip(x) if isinstance(x, str) else x)
        df_meta.set_index("VoxCeleb1 ID", inplace=True)
        return df_meta

    def _get_langs_est(self):
        file_name = "lang_vox2_final.csv"
        file_path = self.corpus_dir / file_name
        if not file_path.exists():
            file_path = self.output_dir / file_name
            if not file_path.exists():
                url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/lang_vox1_final.csv"
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
    def make_cat_list(lists_cat_dir, rec_id, rec_files, video_idx, i):
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

    def make_trials(self):
        url_base = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta"
        trials_file_names = [
            "veri_test2.txt",
            "list_test_hard2.txt",
            "list_test_all2.txt",
        ]
        trials_names = ["trials_o", "trials_h", "trials_e"]

        trials = {}
        dfs = []
        logging.info("making trials")
        for trial_name, file_name in zip(trials_names, trials_file_names):
            file_path = self.corpus_dir / file_name
            if not file_path.exists():
                file_path = self.output_dir / file_name
                if not file_path.exists():
                    url = f"{url_base}/{file_name}"
                    file_path, _ = urlretrieve_progress(url, file_path, desc=file_name)

            df_in = pd.read_csv(
                file_path,
                header=None,
                sep=" ",
                names=["key", "enroll_file", "test_file"],
            )
            key = ["target" if k == 1 else "nontarget" for k in df_in["key"]]

            def get_modelid(s):
                s = re.sub(r"\.wav", "", s)
                return re.sub(r"/", "-", s)

            if self.use_kaldi_ids:
                get_segmentid = get_modelid
            else:

                def get_segmentid(s):
                    s = get_modelid(s)
                    return re.sub(r"[^-]*-", "", s)

            modelid = [get_modelid(f) for f in df_in["enroll_file"]]
            segmentid = [get_segmentid(f) for f in df_in["test_file"]]
            df_out = pd.DataFrame(
                {"modelid": modelid, "segmentid": segmentid, "targettype": key}
            )
            df_out.sort_values(by=["modelid", "segmentid"], inplace=True)
            file_path = self.output_dir / f"{trial_name}.csv"
            df_out.to_csv(file_path, index=False)
            dfs.append(df_out)
            trials[trial_name] = file_path

        df_out = pd.concat(dfs, ignore_index=True)
        df_out.sort_values(by=["modelid", "segmentid"], inplace=True)
        file_path = self.output_dir / "trials.csv"
        df_out.to_csv(file_path, index=False)
        trials["trials"] = file_path

        logging.info("making enrollment map")
        modelid = df_out["modelid"].sort_values().unique()
        if self.use_kaldi_ids:
            segmentid = modelid
        else:
            segmentid = [re.sub(r"[^-]*-", "", s) for s in modelid]

        df_out = pd.DataFrame({"modelid": modelid, "segmentid": segmentid})
        file_path = self.output_dir / "enrollment.csv"
        df_out.to_csv(file_path, index=False)
        enrollments = {"enrollment": file_path}

        return enrollments, trials

    def prepare(self):
        logging.info(
            "Peparing VoxCeleb1 for %s corpus_dir:%s -> data_dir:%s",
            self.task,
            self.corpus_dir,
            self.output_dir,
        )
        logging.info("getting audio meta-data")
        df_meta = self._get_metadata()
        logging.info("getting language estimations")
        df_lang = self._get_langs_est()
        rec_dir = self.corpus_dir
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.wav"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.wav", recursive=True)
            ]

        speakers = [f.parents[1].name for f in rec_files]
        video_ids = [f.parent.name for f in rec_files]
        if self.cat_videos:
            lists_cat_dir = self.output_dir / "lists_cat"
            lists_cat_dir.mkdir(exist_ok=True, parents=True)
            uniq_video_ids, uniq_video_idx, video_idx = np.unique(
                video_ids, return_index=True, return_inverse=True
            )
            rec_ids = uniq_video_ids
            speakers = [speakers[i] for i in uniq_video_idx]
            rec_ids = [f"{s}-{v}" for s, v in zip(speakers, uniq_video_ids)]

            file_paths = []
            futures = []
            logging.info("making video cat lists")
            logging.info("submitting threats...")
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                for i, rec_id in tqdm(enumerate(rec_ids)):
                    future = pool.submit(
                        VoxCeleb1DataPrep.make_cat_list,
                        lists_cat_dir,
                        rec_id,
                        rec_files,
                        video_idx,
                        i,
                    )
                    futures.append(future)

            logging.info("waiting threats...")
            file_paths = [f.result() for f in tqdm(futures)]
            video_ids = uniq_video_ids

        else:
            file_names = [f.with_suffix("").name for f in rec_files]
            if self.use_kaldi_ids:
                rec_ids = [
                    f"{s}-{v}-{f}" for s, v, f in zip(speakers, video_ids, file_names)
                ]
            else:
                rec_ids = [f"{v}-{f}" for v, f in zip(video_ids, file_names)]

            file_paths = [str(r) for r in rec_files]

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
                "video_ids": video_ids,
                "speaker": speakers,
                "gender": df_meta.loc[speakers, "Gender"],
                "nationality": df_meta.loc[speakers, "Nationality"],
                "language_est": [
                    df_lang.loc[r, "lang"] if r in df_lang.index else "N/A"
                    for r in rec_ids
                ],
                "language_est_conf": [
                    df_lang.loc[r, "confidence"] if r in df_lang.index else "N/A"
                    for r in rec_ids
                ],
                "duration": recs.loc[rec_ids, "duration"].values,
            }
        )
        segments = SegmentSet(segments)
        segments.sort()

        logging.info("making speaker info file")
        uniq_speakers = np.unique(speakers)
        speakers = pd.DataFrame(
            {
                "id": uniq_speakers,
                "vgg_id": df_meta.loc[uniq_speakers, "VGGFace1 ID"],
                "gender": df_meta.loc[uniq_speakers, "Gender"],
                "nationality": df_meta.loc[uniq_speakers, "Nationality"],
            }
        )
        speakers = ClassInfo(speakers)

        logging.info("making language info file")
        languages = np.unique(df_lang["lang"])
        languages = ClassInfo(pd.DataFrame({"id": languages}))

        if self.task == "test":
            enrollments, trials = self.make_trials()

        logging.info("making dataset")
        dataset = Dataset(
            segments,
            classes={"speaker": speakers, "language_est": languages},
            recordings={"recordings": recs},
            enrollments=enrollments,
            trials=trials,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )
