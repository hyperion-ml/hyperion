"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .info_table import InfoTable


class ClassInfo(InfoTable):
    def __init__(self, df):
        super().__init__(df)
        if "class_idx" not in self.df:
            self.add_class_idx()

        if "weights" not in self.df:
            self.add_equal_weights()
        else:
            self.df['weights'] /= self.df['weigths'].sum()

    def add_class_idx(self):
        self.df["class_idx"] = [i for i in range(len(self.df))]

    def add_equal_weights(self):
        self.df["weights"] = 1 / len(self.df)

    @property
    def weights(self, id):
        return self.df.loc[id, "weights"]
