"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable

# import torchvision

T = TypeVar("T", bound="ImageSet")


class ImageSet(InfoTable):
    """
    InfoTable specialization for image manifests.

    The table must contain ``id`` and ``storage_path`` columns.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.image_set import ImageSet
        >>> df = pd.DataFrame({"id": ["img1"], "storage_path": ["a/b/c.jpg"]})
        >>> images = ImageSet(df)
        >>> images.df.loc["img1", "storage_path"]
        'a/b/c.jpg'
        >>> images2 = images.filter(items=["img1"])
        >>> list(images2.index)
        ['img1']
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize an image set.

        Args:
            df (pd.DataFrame or ImageSet): Input metadata table.
        """
        super().__init__(df)
        assert "storage_path" in df

    # @staticmethod
    # def _get_durations(videos, i, n):

    #     videos = videos.split(i, n)
    #     durations = []
    #     fss = []
    #     fpss = []
    #     for i, video in videos.iterrows():
    #         reader = torchvision.io.VideoReader(video["storage_path"], "video")
    #         # The information about the video can be retrieved using the
    #         # `get_metadata()` method. It returns a dictionary for every stream, with
    #         # duration and other relevant metadata (often frame rate)
    #         reader_md = reader.get_metadata()

    #         # metadata is structured as a dict of dicts with following structure
    #         # {"stream_type": {"attribute": [attribute per stream]}}
    #         #
    #         # following would print out the list of frame rates for every present video stream
    #         print(reader_md)

    #     return fss, fpss, durations

    # def get_durations(self, num_threads: int = 16):

    #     import itertools
    #     from concurrent.futures import ThreadPoolExecutor

    #     from tqdm import tqdm

    #     futures = []
    #     num_threads = min(num_threads, len(self.df))
    #     logging.info("submitting threats...")
    #     with ThreadPoolExecutor(max_workers=num_threads) as pool:
    #         for i in tqdm(range(num_threads)):
    #             future = pool.submit(VideoSet._get_durations, self, i, num_threads)
    #             futures.append(future)

    #     logging.info("waiting threats...")
    #     res = [f.result() for f in tqdm(futures)]
    #     fss = list(itertools.chain(*[r[0] for r in res]))
    #     durations = list(itertools.chain(*[r[1] for r in res]))

    #     self.df["duration"] = durations
    #     self.df["sample_freq"] = fss
    #     self.df["fps"] = fpss
