"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .info_table import InfoTable


class SegmentSet(InfoTable):
    def __init__(self, df):
        super().__init__(df)

    def recording_ids(self, ids):
        if "recording_id" in self.df:
            return self.df.loc[ids, "recording_id"]

        return ids
