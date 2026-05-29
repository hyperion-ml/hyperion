"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from typing import Optional

import numpy as np

_FLOAT_CPU_ENV = "HYPERION_FLOAT_CPU"
_FLOAT_GPU_ENV = "HYPERION_FLOAT_GPU"
_FLOAT_SAVE_ENV = "HYPERION_FLOAT_SAVE"

_DEFAULT_FLOAT_CPU = "float64"
_DEFAULT_FLOAT_GPU = "float32"
_DEFAULT_FLOAT_SAVE = "float32"


def float_cpu() -> str:
    return os.environ.get(_FLOAT_CPU_ENV, _DEFAULT_FLOAT_CPU)


def set_float_cpu(float_cpu: Optional[str]) -> None:
    if float_cpu is None:
        os.environ.pop(_FLOAT_CPU_ENV, None)
    else:
        os.environ[_FLOAT_CPU_ENV] = str(float_cpu)


def float_gpu() -> str:
    return os.environ.get(_FLOAT_GPU_ENV, _DEFAULT_FLOAT_GPU)


def set_float_gpu(float_gpu: Optional[str]) -> None:
    if float_gpu is None:
        os.environ.pop(_FLOAT_GPU_ENV, None)
    else:
        os.environ[_FLOAT_GPU_ENV] = str(float_gpu)


def float_save() -> str:
    return os.environ.get(_FLOAT_SAVE_ENV, _DEFAULT_FLOAT_SAVE)


def set_float_save(float_save: Optional[str]) -> None:
    if float_save is None:
        os.environ.pop(_FLOAT_SAVE_ENV, None)
    else:
        os.environ[_FLOAT_SAVE_ENV] = str(float_save)


logging_levels = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG, 3: 5}


def config_logger(verbose_level: int) -> None:

    logging_level = logging_levels[verbose_level]
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
