"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from dataclasses import dataclass


@dataclass
class HypDataClass:
    """Dataclass that can imitate a dict"""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        return setattr(self, key, val)

    def keys(self):
        return self.__dict__.keys()
        #return self.__annotations__.keys()

    def items(self):
        return self.__dict__.items()
        # for k in self.keys():
        #     yield k, getattr(self, k)

    @classmethod
    def from_parent(cls, parent, **kwargs):
        args = parent.__dict__
        args.update(kwargs)
        return cls(**args)
