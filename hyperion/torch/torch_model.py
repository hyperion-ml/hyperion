"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict
from copy import deepcopy
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

torch_model_registry = {}


class TorchModel(nn.Module):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        torch_model_registry[cls.__name__] = cls

    def __init__(self):
        super().__init__()
        self._train_mode = "full"

    def get_config(self):
        config = {"class_name": self.__class__.__name__}
        return config

    def copy(self):
        return deepcopy(self)

    def clone(self):
        return deepcopy(self)

    def trainable_parameters(self, recurse: bool = True):
        for param in self.parameters(recurse=recurse):
            if param.requires_grad:
                yield param

    def non_trainable_parameters(self, recurse: bool = True):
        for param in self.parameters(recurse=recurse):
            if not param.requires_grad:
                yield param

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def change_dropouts(self, dropout_rate):
        """Changes all dropout rates of the model."""
        for module in self.modules():
            if isinstance(module, nn.modules.dropout._DropoutNd):
                module.p = dropout_rate

        if hasattr(self, "dropout_rate"):
            assert dropout_rate == 0 or self.dropout_rate > 0
            self.dropout_rate = dropout_rate

    @property
    def train_mode(self):
        return self._train_mode

    @train_mode.setter
    def train_mode(self, mode):
        print("hola3", mode, flush=True)
        self.set_train_mode(mode)

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()

        self._train_mode = mode

    def _train(self, train_mode: str):
        if train_mode == "full":
            super().train(True)
        elif train_mode == "frozen":
            super().train(False)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def train(self, mode: bool = True):
        if not mode:
            super().train(False)
            return

        self._train(self.train_mode)

    @staticmethod
    def valid_train_modes():
        return ["full", "frozen"]

    def save(self, file_path):
        file_dir = os.path.dirname(file_path)
        if not (os.path.isdir(file_dir)):
            os.makedirs(file_dir, exist_ok=True)

        config = self.get_config()
        torch.save(
            {"model_cfg": self.get_config(), "model_state_dict": self.state_dict()}
        )

    @staticmethod
    def _load_cfg_state_dict(file_path=None, cfg=None, state_dict=None):
        model_data = None
        if cfg is None or state_dict is None:
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

    @staticmethod
    def _fix_cfg_compatibility(class_obj, cfg):
        """Function that fixed compatibility issues with deprecated models

        Args:
          class_obj: class type of the model.
          cfg: configuration dictiory that inits the model.

        Returns:
          Fixed configuration dictionary.
        """
        # for compatibility with older x-vector models
        XVector = torch_model_registry["xvector"]
        if issubclass(class_obj, XVector):
            # We renamed AM-softmax scale parameer s to cos_scale
            if "s" in cfg:
                cfg["cos_scale"] = cfg["s"]
                del cfg["s"]

        return cfg

    @staticmethod
    def auto_load(file_path, extra_objs={}, map_location=None):

        if map_location is None:
            map_location = torch.device("cpu")

        model_data = torch.load(file_path, map_location=map_location)
        cfg = model_data["model_cfg"]
        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name in torch_model_registry:
            class_obj = torch_model_registry[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception("unknown object with class_name=%s" % (class_name))

        state_dict = model_data["model_state_dict"]

        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        cfg = TorchModel._fix_cfg_compatibility(class_obj, cfg)

        import re

        p = re.compile("^module\.")
        num_tries = 3
        for tries in range(num_tries):
            try:
                return class_obj.load(cfg=cfg, state_dict=state_dict)
            except RuntimeError as err:
                # remove module prefix when is trained with dataparallel
                if tries == num_tries - 1:
                    # if it failed the 3 trials raise exception
                    raise err
                # remove module prefix when is trained with dataparallel
                state_dict = ODict((p.sub("", k), v) for k, v in state_dict.items())
