"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .info_table import InfoTable


class VideoSet(InfoTable):
    def __init__(self, df):
        super().__init__(df)
        assert "storage_path" in df

    @staticmethod
    def _get_metadata(videos, i, n):
        import av

        videos = videos.split(i, n)
        durations = []
        video_durations = []
        fss = []
        fpss = []
        for i, video in videos.iterrows():

            with av.open(str(video["storage_path"])) as f:
                # fps = (
                #     float(f.streams.video[0].average_rate.numerator)
                #     / f.streams.video[0].average_rate.denominator
                # )
                fps = float(f.streams.video[0].average_rate)
                fs = f.streams.audio[0].sample_rate
                if f.streams.audio[0].duration is not None:
                    audio_duration = float(
                        f.streams.audio[0].duration * f.streams.audio[0].time_base
                    )
                else:
                    audio_duration = 0.0

                if f.streams.video[0].duration is not None:
                    video_duration = float(
                        f.streams.video[0].duration * f.streams.video[0].time_base
                    )
                else:
                    video_duration = 0.0

                fpss.append(fps)
                fss.append(fs)
                durations.append(audio_duration)
                video_durations.append(video_duration)

            # reader = torchvision.io.VideoReader(video["storage_path"], "video")
            # # The information about the video can be retrieved using the
            # # `get_metadata()` method. It returns a dictionary for every stream, with
            # # duration and other relevant metadata (often frame rate)
            # reader_md = reader.get_metadata()

            # metadata is structured as a dict of dicts with following structure
            # {"stream_type": {"attribute": [attribute per stream]}}
            #
            # following would print out the list of frame rates for every present video stream
            # print(reader_md)

        return fss, durations, fpss, video_durations

    def get_metadata(self, num_threads: int = 16):

        import itertools
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        futures = []
        num_threads = min(num_threads, len(self.df))
        logging.info("submitting threats...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for i in tqdm(range(num_threads)):
                future = pool.submit(VideoSet._get_metadata, self, i, num_threads)
                futures.append(future)

        logging.info("waiting threats...")
        res = [f.result() for f in tqdm(futures)]
        fss = list(itertools.chain(*[r[0] for r in res]))
        durations = list(itertools.chain(*[r[1] for r in res]))
        fpss = list(itertools.chain(*[r[2] for r in res]))
        vid_durations = list(itertools.chain(*[r[3] for r in res]))

        self.df["duration"] = durations
        self.df["video_duration"] = vid_durations
        self.df["sample_freq"] = fss
        self.df["fps"] = fpss
