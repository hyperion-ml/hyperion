"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from collections import OrderedDict as ODict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from ..utils.misc import PathLike


class TorchModel(nn.Module):
    """Base class for all Pytorch Models and NNet architectures"""

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        TorchModel.registry[cls.__name__] = cls

    def __init__(self, bias_weight_decay=None):
        super().__init__()
        self._train_mode = "full"
        self.bias_weight_decay = bias_weight_decay

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

    def trainable_named_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad:
                yield name, param

    def non_trainable_named_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if not param.requires_grad:
                yield name, param

    def parameter_summary(self, verbose: bool = False):
        trainable_params = sum(p.numel() for p in self.trainable_parameters())
        non_trainable_params = sum(p.numel() for p in self.non_trainable_parameters())
        buffer_params = sum(p.numel() for p in self.buffers())
        non_trainable_total = non_trainable_params + buffer_params
        total_params = trainable_params + non_trainable_total
        if verbose:
            logging.info(
                "total-params=%d, trainable-params=%d, non-trainable-params+buffers=%d, non-trainable-params=%d, buffer-params=%d",
                total_params,
                trainable_params,
                non_trainable_total,
                non_trainable_params,
                buffer_params,
            )
        return (
            total_params,
            trainable_params,
            non_trainable_total,
            non_trainable_params,
            buffer_params,
        )

    def print_parameter_list(self):
        for n, p in self.trainable_named_parameters():
            logging.info("trainable: %s", n)

        for n, p in self.non_trainable_named_parameters():
            logging.info("non_trainable: %s", n)

        for n, p in self.named_buffers():
            logging.info("buffers: %s", n)

    def has_param_groups(self):
        return self.bias_weight_decay is not None

    def trainable_param_groups(self):
        if self.bias_weight_decay is None:
            return [{"params": self.trainable_parameters()}]

        regularized = []
        not_regularized = []
        for name, param in self.trainable_named_parameters():
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)

        return [
            {"params": regularized},
            {"params": not_regularized, "weight_decay": self.bias_weight_decay},
        ]

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
            if isinstance(module, nn.RNNBase):
                module.dropout = dropout_rate

        if hasattr(self, "dropout_rate"):
            assert dropout_rate == 0 or self.dropout_rate > 0
            self.dropout_rate = dropout_rate

    @property
    def train_mode(self):
        return self._train_mode

    @train_mode.setter
    def train_mode(self, mode):
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
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_cfg": self.get_config(), "model_state_dict": self.state_dict()},
            file_path,
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
    def _remove_module_prefix(state_dict):
        import re

        p = re.compile("^(module\.)+")
        if p.match(list(state_dict.keys())[0]) is not None:
            state_dict = ODict((p.sub("", k), v) for k, v in state_dict.items())

        return state_dict

    @staticmethod
    def _fix_xvector_cfg(cfg):
        # We renamed AM-softmax scale parameer s to cos_scale
        if "s" in cfg:
            cfg["cos_scale"] = cfg.pop("s")

        return cfg

    @staticmethod
    def _fix_hf_wav2xvector(cfg, state_dict):
        key = "feat_fusion_method"
        if key in cfg:
            fuser_type = cfg.pop(key)
            feat_fuser = {
                "feat_fuser": {"fuser_type": fuser_type},
                "mvn": None,
                "spec_augment": None,
            }
            cfg["feat_fuser"] = feat_fuser
            state_dict["feat_fuser.feat_fuser.feat_fuser"] = state_dict.pop(
                "feat_fuser"
            )

        return cfg, state_dict

    @staticmethod
    def _fix_model_compatibility(class_obj, cfg, state_dict):
        """Function that fixed compatibility issues with deprecated models

        Args:
          class_obj: class type of the model.
          cfg: configuration dictiory that inits the model.

        Returns:
          Fixed configuration dictionary.
        """
        # for compatibility with older x-vector models
        XVector = TorchModel.registry["XVector"]
        if issubclass(class_obj, XVector):
            cfg = TorchModel._fix_xvector_cfg(cfg)

        # switch old feature fuser to new feature fuser in w2v x-vectors
        HFWav2XVector = TorchModel.registry["HFWav2XVector"]
        if issubclass(class_obj, HFWav2XVector):
            cfg, state_dict = TorchModel._fix_hf_wav2xvector(cfg, state_dict)

        return cfg, state_dict

    @staticmethod
    def _is_hf_path(file_path: Path):
        # hf path can have only 2 dir levels
        return len(file_path.parents) == 2

    @staticmethod
    def _get_from_hf(
        file_path: Path, cache_dir: PathLike = None, local_dir: PathLike = None
    ):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=file_path.parent,
            filename=file_path.name,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )

    @staticmethod
    def _try_to_get_from_hf(
        file_path: Path, cache_dir: PathLike = None, local_dir: PathLike = None
    ):
        if str(file_path)[:3] == "hf:":
            # hf: prefix indicates to download from hub
            file_path = Path(str(file_path)[3:])
            assert TorchModel._is_hf_path(
                file_path
            ), f"{file_path} is not a valid HF path"
            file_path = TorchModel._get_from_hf(
                file_path, cache_dir=cache_dir, local_dir=local_dir
            )
            return Path(file_path)
        elif not file_path.is_file():
            # if no prefix but file not in local dir try to get it from hub
            if not TorchModel._is_hf_path(file_path):
                return file_path

            try:
                file_path = TorchModel._get_from_hf(file_path)
                return Path(file_path)
            except:
                return file_path

        else:
            # file is local
            return file_path

    @staticmethod
    def auto_load(
        file_path: PathLike,
        model_name: Optional[str] = None,
        extra_objs: dict = {},
        map_location: Optional[
            Union[
                Callable[[torch.Tensor, str], torch.Tensor],
                torch.device,
                str,
                Dict[str, str],
            ]
        ] = None,
        cache_dir: PathLike = None,
        local_dir: PathLike = None,
    ):
        file_path = Path(file_path)
        file_path = TorchModel._try_to_get_from_hf(
            file_path, cache_dir=cache_dir, local_dir=local_dir
        )

        assert file_path.is_file(), f"TorchModel file: {file_path} not found"

        if map_location is None:
            map_location = torch.device("cpu")

        model_data = torch.load(file_path, map_location=map_location)
        cfg = model_data["model_cfg"]
        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name in TorchModel.registry:
            class_obj = TorchModel.registry[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception("unknown object with class_name=%s" % (class_name))

        if model_name is None:
            model_name = "model"
        state_dict = model_data[f"{model_name}_state_dict"]

        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        state_dict = TorchModel._remove_module_prefix(state_dict)
        cfg, state_dict = TorchModel._fix_model_compatibility(
            class_obj, cfg, state_dict
        )

        return class_obj.load(cfg=cfg, state_dict=state_dict)
        # num_tries = 3
        # for tries in range(num_tries):
        #     try:
        #         return class_obj.load(cfg=cfg, state_dict=state_dict)
        #     except RuntimeError as err:
        #         # remove module prefix when is trained with dataparallel
        #         if tries == num_tries - 1:
        #             # if it failed the 3 trials raise exception
        #             raise err
        #         # remove module prefix when is trained with dataparallel
        #         state_dict = ODict((p.sub("", k), v) for k, v in state_dict.items())
