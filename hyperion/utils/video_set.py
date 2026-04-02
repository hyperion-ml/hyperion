"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable

T = TypeVar("T", bound="VideoSet")


class VideoSet(InfoTable):
    """
    InfoTable specialization for audiovisual recording manifests.

    The table must contain ``id`` and ``storage_path`` columns.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.video_set import VideoSet
        >>> df = pd.DataFrame({"id": ["vid1"], "storage_path": ["/video/vid1.mp4"]})
        >>> videos = VideoSet(df)
        >>> videos.df.loc["vid1", "storage_path"]
        '/video/vid1.mp4'
        >>> videos2 = videos.filter(items=["vid1"])
        >>> len(videos2)
        1
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize a video set.

        Args:
            df (pd.DataFrame or VideoSet): Input metadata table.
        """
        super().__init__(df)
        assert "storage_path" in df

    @staticmethod
    def _get_metadata(
        videos: "VideoSet", i: int, n: int
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """
        Collect audio/video metadata for one data partition.

        Args:
            videos (VideoSet): Source table.
            i (int): 1-based partition index.
            n (int): Number of partitions.

        Returns:
            Tuple[List[int], List[float], List[float], List[float]]:
            Sample rates, audio durations, FPS values, and video durations.
        """
        import av

        videos = videos.split(i, n)
        durations = []
        video_durations = []
        fss = []
        fpss = []
        for i, video in videos.iterrows():

            with av.open(str(video["storage_path"])) as f:
                video_stream = f.streams.video[0] if len(f.streams.video) > 0 else None
                audio_stream = f.streams.audio[0] if len(f.streams.audio) > 0 else None

                # fps = (
                #     float(f.streams.video[0].average_rate.numerator)
                #     / f.streams.video[0].average_rate.denominator
                # )
                if video_stream is not None and video_stream.average_rate is not None:
                    fps = float(video_stream.average_rate)
                else:
                    fps = 0.0

                if audio_stream is not None and audio_stream.sample_rate is not None:
                    fs = int(audio_stream.sample_rate)
                else:
                    fs = 0

                if (
                    audio_stream is not None
                    and audio_stream.duration is not None
                    and audio_stream.time_base is not None
                ):
                    audio_duration = float(
                        audio_stream.duration * audio_stream.time_base
                    )
                else:
                    audio_duration = 0.0

                if (
                    video_stream is not None
                    and video_stream.duration is not None
                    and video_stream.time_base is not None
                ):
                    video_duration = float(
                        video_stream.duration * video_stream.time_base
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

    def get_metadata(self, num_threads: int = 16) -> None:
        """
        Populate duration and frame-rate metadata using a thread pool.

        Args:
            num_threads (int): Maximum number of worker threads.
        """

        import itertools
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        total = len(self.df)
        if total == 0:
            logging.info("No videos available; skipping metadata extraction.")
            return

        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        futures = []
        num_threads = min(num_threads, total)
        logging.info("submitting threats...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for i in tqdm(range(num_threads)):
                future = pool.submit(VideoSet._get_metadata, self, i + 1, num_threads)
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
