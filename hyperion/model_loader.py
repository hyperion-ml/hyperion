"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .hyp_model import HypModel
from .pdfs import *
from .transforms import *


class ModelLoader(object):
    @staticmethod
    def get_object():
        obj_dict = {
            "DiagNormal": DiagNormal,
            "Normal": Normal,
            "DiagGMM": DiagGMM,
            "GMM": GMM,
            "FRPLDA": FRPLDA,
            "SPLDA": SPLDA,
            "CentWhiten": CentWhiten,
            "LNorm": LNorm,
            "PCA": PCA,
            "LDA": LDA,
            "NAP": NAP,
            "SbSw": SbSw,
            "MVN": MVN,
            "TransformList": TransformList,
        }
        return obj_dict

    @staticmethod
    def load(file_path):
        class_name = HypModel.load_config(file_path)["class_name"]
        class_obj = ModelLoader.get_object()[class_name]
        return class_obj.load(file_path)
