"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable


class SegmentSet(InfoTable):
    """Class to store information about a speech segment
    Internally, it uses a pandas table.
    """

    def __init__(self, df):
        super().__init__(df)
        if "start" in df and "recording" not in df:
            df["recording"] = df["id"]
            df.fillna(values={"start": 0.0}, inplace=True)

        if "start" not in df and "recording" in df:
            df["start"] = 0.0

        if "recording" in df:
            is_nan = df["recording"].isnan()
            df.loc[is_nan, "recording"] = df.loc[is_nan, "id"]

    @property
    def has_time_marks(self):
        return "recording" in self.df and "start" in self.df and "duration" in self.df

    @property
    def has_recording_ids(self):
        return "recording" in self.df

    @property
    def has_recording(self):
        return "recording" in self.df

    def recording(self, ids: Union[np.ndarray, List[str], None] = None):
        if ids is None:
            if "recording" in self.df:
                return self.df["recording"]
            else:
                return self.df["id"]

        if "recording" in self.df:
            return self.df.loc[ids, "recording"]

        return ids

    def image(self, ids: Union[np.ndarray, List[str], None] = None):
        if ids is None:
            if "image" in self.df:
                return self.df["image"]
            else:
                return self.df["id"]

        if "image" in self.df:
            return self.df.loc[ids, "image"]

        return ids

    def video(self, ids: Union[np.ndarray, List[str], None] = None):
        if ids is None:
            if "video" in self.df:
                return self.df["video"]
            else:
                return self.df["video"]

        if "video" in self.df:
            return self.df.loc[ids, "video"]

        return ids

    def recording_ids(self, ids: Union[np.ndarray, List[str], None] = None):
        return self.recording(ids)

    def recording_time_marks(self, ids: Union[np.ndarray, List[str]]):
        if "recording" in self.df:
            recording_name = "recording"
        else:
            recording_name = "id"

        assert "duration" in self.df
        if "start" not in self.df:
            self.df["start"] = 0.0

        return self.df.loc[ids, [recording_name, "start", "duration"]]

    def sample_random_subsegments(
        self,
        subsegments_per_segment: int = 1,
        min_duration: float = 0.0,
        max_duration: Optional[float] = None,
        seg_suffix: Optional[str] = None,
        random_start: bool = True,
        seed: int = 11235813,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng(seed)

        dfs = []
        for i in range(subsegments_per_segment):
            if max_duration is None:
                duration = rng.uniform(
                    low=min_duration, high=self.df["duration"].values
                )
            else:
                duration = rng.uniform(
                    low=min_duration, high=max_duration, size=(len(self.df),)
                )
                duration = np.minimum(duration, self.df["duration"].values)

            if random_start:
                t_start = rng.uniform(
                    low=0.0, high=self.df["duration"].values - duration
                )
            else:
                t_start = 0.0

            df = self.df.copy()
            df["start"] = t_start
            df["duration"] = duration
            if seg_suffix is None:
                suffix_i = f"-{i}" if subsegments_per_segment > 1 else None
            else:
                suffix_i = (
                    f"{seg_suffix}-{i}" if subsegments_per_segment > 1 else seg_suffix
                )

            if suffix_i is not None:
                df["id"] = df["id"].apply(lambda x: f"{x}-{suffix_i}")

            dfs.append(df)

        df = pd.concat(dfs)
        return SegmentSet(df)
