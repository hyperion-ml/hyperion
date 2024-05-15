"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path

import yaml

from ...utils.misc import PathLike


class HypTokenizer:
    """Base class for tokenizers in Hyperion"""

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        HypTokenizer.registry[cls.__name__] = cls

    def normalize(self, text):
        return text

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    @staticmethod
    def auto_load(file_path: PathLike):
        file_path = Path(file_path)
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)

        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name in HypTokenizer.registry:
            class_obj = HypTokenizer.registry[class_name]
        else:
            raise Exception("unknown object with class_name=%s" % (class_name))

        return class_obj.load(file_path)
