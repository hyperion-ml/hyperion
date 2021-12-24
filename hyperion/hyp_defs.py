"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging

import numpy as np

_FLOAT_CPU = "float64"
_FLOAT_KERAS = "float32"
_FLOAT_SAVE = "float32"


def float_cpu():
    return _FLOAT_CPU


def set_float_cpu(float_cpu):
    global _FLOAT_CPU
    _FLOAT_CPU = float_cpu


def float_keras():
    return _FLOAT_KERAS


def set_float_keras(float_keras):
    global _FLOAT_KERAS
    _FLOAT_KERAS = float_keras


def float_save():
    return _FLOAT_SAVE


def set_float_save(float_save):
    global _FLOAT_SAVE
    _FLOAT_SAVE = float_save


logging_levels = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG, 3: 5}


def config_logger(verbose_level):

    logging_level = logging_levels[verbose_level]
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
