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

    def recording(self, ids=None):
        if ids is None:
            if "recording" in self.df:
                return self.df["recording"]
            else:
                return self.df["id"]

        if "recording" in self.df:
            return self.df.loc[ids, "recording"]

        return ids

    def recording_ids(self, ids=None):
        return self.recording(ids)

    def recording_time_marks(self, ids):
        if "recording" in self.df:
            recording_name = "recording"
        else:
            recording_name = "id"

        assert "duration" in self.df
        if "start" not in self.df:
            self.df["start"] = 0.0

        return self.df.loc[ids, [recording_name, "start", "duration"]]
