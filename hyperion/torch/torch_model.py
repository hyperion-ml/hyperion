"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from copy import deepcopy

import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def get_config(self):
        config = {"class_name": self.__class__.__name__}

        return config

    def copy(self):
        return deepcopy(self)

    def save(self, file_path):
        file_dir = os.path.dirname(file_path)
        if not (os.path.isdir(file_dir)):
            os.makedirs(file_dir, exist_ok=True)

        config = self.get_config()
        torch.save(
            {"model_cfg": self.get_config(), "model_state_dict": self.state_dict()}
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @staticmethod
    def _load_cfg_state_dict(file_path=None, cfg=None, state_dict=None):
        model_data = None
        if cfg is None:
            assert file_path is not None
            model_data = torch.load(file_path)
        if cfg is None:
            cfg = model_data["model_cfg"]
        if state_dict is None and model_data is not None:
            state_dict = model_data["model_state_dict"]

        if "class_name" in cfg:
            del cfg["class_name"]

        return cfg, state_dict

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = TorchModel._load_cfg_state_dict(file_path, cfg, state_dict)

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def get_reg_loss(self):
        return 0

    def get_loss(self):
        return 0

    @property
    def device(self):
        devices = {param.device for param in self.parameters()} | {
            buf.device for buf in self.buffers()
        }
        if len(devices) != 1:
            raise RuntimeError(
                "Cannot determine device: {} different devices found".format(
                    len(devices)
                )
            )

        return next(iter(devices))
