"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from pathlib import Path

import numpy as np
import pandas as pd



def read_text(text_file: str):
    # assert check_argument_types()
    text_file = Path(text_file)

    data = {"id":[],"text":[]}
    with Path(text_file).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            # if k in data:
            #     raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data["id"].append(k)
            data["text"].append(v)
    return pd.DataFrame(data=data, index=data["id"])

