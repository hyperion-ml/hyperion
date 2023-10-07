"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .info_table import InfoTable


class SegmentSet(InfoTable):
    """Class to store information about a speech segment
    Internally, it uses a pandas table.
    """

    def __init__(self, df):
        super().__init__(df)
        if "start" in df and "recordings" not in df:
            df["recordings"] = df["id"]

        if "start" not in df and "recordings" in df:
            df["start"] = 0.0

    @property
    def has_time_marks(self):
        return "recordings" in self.df and "start" in self.df and "duration" in self.df

    @property
    def has_recording_ids(self):
        return "recordings" in self.df

    @property
    def has_recordings(self):
        return "recordings" in self.df

    def recordings(self, ids=None):
        if ids is None:
            if "recordings" in self.df:
                return self.df["recordings"]
            else:
                return self.df["id"]

        if "recordings" in self.df:
            return self.df.loc[ids, "recordings"]

        return ids

    def recording_ids(self, ids=None):
        return self.recordings(ids)

    def recording_time_marks(self, ids, recordings_name: str = "recordings"):
        if recordings_name == "recordings":
            if "recordings" in self.df:
                recordings_name = "recordings"
            else:
                recordings_name = "id"

        assert "duration" in self.df
        if "start" not in self.df:
            self.df["start"] = 0.0

        return self.df.loc[ids, [recordings_name, "start", "duration"]]
