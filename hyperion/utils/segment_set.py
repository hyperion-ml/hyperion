"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable


class SegmentSet(InfoTable):
    """
    Store metadata for speech segments.

    The table uses ``id`` as segment identifier and may include columns such as
    ``recording``, ``start``, ``duration``, ``image``, and ``video``.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.segment_set import SegmentSet
        >>> df = pd.DataFrame({"id": ["seg1"], "recording": ["rec1"], "duration": [1.2]})
        >>> segs = SegmentSet(df)
        >>> segs.has_time_marks
        True
        >>> segs.recording(["seg1"]).tolist()
        ['rec1']
        >>> marks = segs.recording_time_marks(["seg1"])
        >>> list(marks.columns)
        ['recording', 'start', 'duration']
    """

    def __init__(self, df: Union[pd.DataFrame, "SegmentSet"]) -> None:
        """
        Initialize a segment set and normalize basic timing columns.

        Args:
            df (pd.DataFrame or SegmentSet): Input segment table.
        """
        super().__init__(df)
        if "start" in df and "recording" not in df:
            df["recording"] = df["id"]
            df.fillna(value={"start": 0.0}, inplace=True)

        if "start" not in df and "recording" in df:
            df["start"] = 0.0

        if "recording" in df:
            is_na = df["recording"].isna()
            df.loc[is_na, "recording"] = df.loc[is_na, "id"]

    @property
    def has_time_marks(self) -> bool:
        """
        Whether recording/time-mark columns are present.

        Returns:
            bool: True when ``recording``, ``start``, and ``duration`` exist.
        """
        return "recording" in self.df and "start" in self.df and "duration" in self.df

    @property
    def has_recording_ids(self) -> bool:
        """
        Whether a ``recording`` column is present.

        Returns:
            bool: True if ``recording`` exists.
        """
        return "recording" in self.df

    @property
    def has_recording(self) -> bool:
        """
        Alias for :meth:`has_recording_ids`.

        Returns:
            bool: True if ``recording`` exists.
        """
        return "recording" in self.df

    def recording(
        self, ids: Union[np.ndarray, List[str], None] = None
    ) -> Union[pd.Series, np.ndarray, List[str]]:
        """
        Get recording IDs for segments.

        Args:
            ids (Union[np.ndarray, List[str], None]): Segment IDs to query. If
            ``None``, return the full recording series.

        Returns:
            Union[pd.Series, np.ndarray, List[str]]: Recording IDs. Falls back to
            segment ``id`` when ``recording`` is missing.
        """
        if ids is None:
            if "recording" in self.df:
                return self.df["recording"]
            else:
                return self.df["id"]

        if "recording" in self.df:
            return self.df.loc[ids, "recording"]

        return ids

    def image(
        self, ids: Union[np.ndarray, List[str], None] = None
    ) -> Union[pd.Series, np.ndarray, List[str]]:
        """
        Get image IDs associated with segments.

        Args:
            ids (Union[np.ndarray, List[str], None]): Segment IDs to query. If
            ``None``, return the full image series.

        Returns:
            Union[pd.Series, np.ndarray, List[str]]: Image IDs. Falls back to
            segment ``id`` when ``image`` is missing.
        """
        if ids is None:
            if "image" in self.df:
                return self.df["image"]
            else:
                return self.df["id"]

        if "image" in self.df:
            return self.df.loc[ids, "image"]

        return ids

    def video(
        self, ids: Union[np.ndarray, List[str], None] = None
    ) -> Union[pd.Series, np.ndarray, List[str]]:
        """
        Get video IDs associated with segments.

        Args:
            ids (Union[np.ndarray, List[str], None]): Segment IDs to query. If
            ``None``, return the full video series.

        Returns:
            Union[pd.Series, np.ndarray, List[str]]: Video IDs.
        """
        if ids is None:
            if "video" in self.df:
                return self.df["video"]
            else:
                return self.df["id"]

        if "video" in self.df:
            return self.df.loc[ids, "video"]

        return ids

    def recording_ids(
        self, ids: Union[np.ndarray, List[str], None] = None
    ) -> Union[pd.Series, np.ndarray, List[str]]:
        """
        Alias for :meth:`recording`.

        Args:
            ids (Union[np.ndarray, List[str], None]): Segment IDs to query.

        Returns:
            Union[pd.Series, np.ndarray, List[str]]: Recording IDs.
        """
        return self.recording(ids)

    def recording_time_marks(self, ids: Union[np.ndarray, List[str]]) -> pd.DataFrame:
        """
        Return recording name, start time, and duration for selected segments.

        Args:
            ids (Union[np.ndarray, List[str]]): Segment IDs to query.

        Returns:
            pd.DataFrame: Columns ``[recording_or_id, start, duration]``.
        """
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
    ) -> "SegmentSet":
        """
        Sample random subsegments from each segment.

        Args:
            subsegments_per_segment (int): Number of subsegments to sample per row.
            min_duration (float): Minimum sampled duration.
            max_duration (Optional[float]): Maximum sampled duration. If ``None``,
            each segment's original duration is used as upper bound.
            seg_suffix (Optional[str]): Optional suffix for generated segment IDs.
            random_start (bool): If True, sample random start offsets; otherwise
            use ``0.0``.
            seed (int): RNG seed used when ``rng`` is ``None``.
            rng (Optional[np.random.Generator]): Optional external RNG.

        Returns:
            SegmentSet: New table containing sampled subsegments.
        """
        if subsegments_per_segment <= 0:
            raise ValueError("subsegments_per_segment must be >= 1")

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
