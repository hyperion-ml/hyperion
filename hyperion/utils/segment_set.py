"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .info_table import InfoTable


class SegmentSet(InfoTable):
    def __init__(self, df):
        super().__init__(df)
        if "start" in df and "recording_id" not in df:
            df["recording_id"] = df["id"]

        if "start" not in df and "recording_id" in df:
            df["start"] = 0.0

    @property
    def has_time_marks(self):
        return (
            "recording_id" in self.df and "start" in self.df and "duration" in self.df
        )

    @property
    def has_recording_ids(self):
        return "recording_id" in self.df

    def recording_ids(self, ids):
        if "recording_id" in self.df:
            return self.df.loc[ids, "recording_id"]

        return ids

    def recording_time_marks(self, ids):
        if "recording" in self.df:
            rec_col = "recording_id"
        else:
            rec_col = "id"

        assert "duration" in self.df
        if "start" not in self.df:
            self.df["start"] = 0.0

        return self.df.loc[ids, [rec_col, "start", "duration"]]
