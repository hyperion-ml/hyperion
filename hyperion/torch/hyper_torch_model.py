"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict as ODict
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from ..utils.misc import PathLike


class HyperTorchModel(nn.Module):
    """Base class for PyTorch models and neural network architectures."""

    registry: Dict[str, Type["HyperTorchModel"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses by class name for dynamic loading.

        Args:
          **kwargs: Additional subclass initialization options.
        """
        super().__init_subclass__(**kwargs)
        HyperTorchModel.registry[cls.__name__] = cls

    def __init__(self, bias_weight_decay: Optional[float] = None):
        """Initialize model-level training controls.

        Args:
          bias_weight_decay: Optional decay value for bias/1D parameter group.
        """
        super().__init__()
        self._train_mode = "full"
        self.bias_weight_decay = bias_weight_decay

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
          no_class_name: If ``True``, omit ``class_name`` from the config.

        Returns:
          Configuration dictionary used to reconstruct the model.
        """
        if no_class_name:
            return {}
        config = {"class_name": self.__class__.__name__}
        return config

    def copy(self) -> "HyperTorchModel":
        """Return a deep copy of this model.

        Returns:
          A deep-copied model instance.
        """
        return deepcopy(self)

    def clone(self) -> "HyperTorchModel":
        """Return a deep copy of this model (alias of ``copy``).

        Returns:
          A deep-copied model instance.
        """
        return deepcopy(self)

    def trainable_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield parameters with ``requires_grad=True``.

        Args:
          recurse: Whether to recurse into submodules.

        Returns:
          Iterator of trainable parameters.
        """
        for param in self.parameters(recurse=recurse):
            if param.requires_grad:
                yield param

    def non_trainable_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield parameters with ``requires_grad=False``.

        Args:
          recurse: Whether to recurse into submodules.

        Returns:
          Iterator of non-trainable parameters.
        """
        for param in self.parameters(recurse=recurse):
            if not param.requires_grad:
                yield param

    def trainable_named_parameters(
        self, recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield ``(name, parameter)`` pairs for trainable parameters.

        Args:
          recurse: Whether to recurse into submodules.

        Returns:
          Iterator of named trainable parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad:
                yield name, param

    def non_trainable_named_parameters(
        self, recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield ``(name, parameter)`` pairs for non-trainable parameters.

        Args:
          recurse: Whether to recurse into submodules.

        Returns:
          Iterator of named non-trainable parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if not param.requires_grad:
                yield name, param

    def parameter_summary(
        self, verbose: bool = False
    ) -> Tuple[int, int, int, int, int]:
        """Return parameter and buffer counts.

        Args:
          verbose: If ``True``, log a summary line.

        Returns:
          Tuple ``(total, trainable, non_trainable_plus_buffers,
          non_trainable, buffers)``.
        """
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

    def print_parameter_list(self) -> None:
        """Log names of trainable, non-trainable, and buffer tensors."""
        for n, p in self.trainable_named_parameters():
            logging.info("trainable: %s", n)

        for n, p in self.non_trainable_named_parameters():
            logging.info("non_trainable: %s", n)

        for n, p in self.named_buffers():
            logging.info("buffers: %s", n)

    def has_param_groups(self) -> bool:
        """Return whether model exposes custom optimizer parameter groups.

        Returns:
          ``True`` if ``bias_weight_decay`` is enabled.
        """
        return self.bias_weight_decay is not None

    def trainable_param_groups(self) -> List[Dict[str, Any]]:
        """Build optimizer parameter groups for trainable parameters.

        Returns:
          Optimizer parameter-group dictionaries.
        """
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

    def freeze(self) -> None:
        """Disable gradients for all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Enable gradients for all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def has_batchnorms(self) -> bool:
        """Return ``True`` if the model contains any batch-normalization layer.

        Returns:
          ``True`` when any batchnorm module is found.
        """
        for module in self.modules():
            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LazyBatchNorm1d,
                    nn.LazyBatchNorm2d,
                    nn.LazyBatchNorm3d,
                    nn.SyncBatchNorm,
                ),
            ):
                return True

        return False

    def change_dropouts(self, dropout_rate: float) -> None:
        """Set dropout probability on dropout and RNN modules.

        Args:
          dropout_rate: New dropout rate.
        """
        for module in self.modules():
            if isinstance(module, nn.modules.dropout._DropoutNd):
                module.p = dropout_rate
            if isinstance(module, nn.RNNBase):
                module.dropout = dropout_rate

        if hasattr(self, "dropout_rate"):
            assert dropout_rate == 0 or self.dropout_rate > 0
            self.dropout_rate = dropout_rate

    @property
    def train_mode(self) -> str:
        """Current train mode: ``full`` or ``frozen``.

        Returns:
          Active train mode string.
        """
        return self._train_mode

    @train_mode.setter
    def train_mode(self, mode: str) -> None:
        self.set_train_mode(mode)

    def set_train_mode(self, mode: str) -> None:
        """Switch model between full-train and frozen-parameter modes.

        Args:
          mode: Target mode (``full`` or ``frozen``).
        """
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()

        self._train_mode = mode

    def _train(self, train_mode: str) -> None:
        """Apply PyTorch train/eval state for a custom train mode.

        Args:
          train_mode: Mode to apply (``full`` or ``frozen``).
        """
        if train_mode == "full":
            super().train(True)
        elif train_mode == "frozen":
            super().train(False)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def train(self, mode: bool = True) -> "HyperTorchModel":
        """Override ``nn.Module.train`` to honor ``self.train_mode``.

        Args:
          mode: If ``False``, force eval mode.

        Returns:
          ``self``.
        """
        if not mode:
            super().train(False)
            return self

        self._train(self.train_mode)
        return self

    @staticmethod
    def valid_train_modes() -> List[str]:
        """Return the list of supported train modes.

        Returns:
          Supported train-mode names.
        """
        return ["full", "frozen"]

    def save(self, file_path: PathLike) -> None:
        """Save model config and state dictionary to disk.

        Args:
          file_path: Destination checkpoint path.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_cfg": self.get_config(), "model_state_dict": self.state_dict()},
            file_path,
        )

    @staticmethod
    def _load_cfg_state_dict(
        file_path: Optional[PathLike] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        """Resolve config/state_dict from args or a checkpoint file.

        Args:
          file_path: Optional checkpoint path.
          cfg: Optional pre-loaded model config.
          state_dict: Optional pre-loaded state dict.

        Returns:
          Tuple ``(cfg, state_dict)`` with ``class_name`` removed from config.
        """
        model_data = None
        if cfg is None or state_dict is None:
            assert file_path is not None
            # PyTorch 2.6 changed torch.load default to weights_only=True.
            # Our checkpoints include non-tensor metadata in model_cfg.
            model_data = torch.load(file_path, weights_only=False)
        if cfg is None:
            cfg = model_data["model_cfg"]
        if state_dict is None and model_data is not None:
            state_dict = model_data["model_state_dict"]

        if "class_name" in cfg:
            del cfg["class_name"]

        return cfg, state_dict

    @classmethod
    def load(
        cls,
        file_path: Optional[PathLike] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "HyperTorchModel":
        """Instantiate model from config and optionally load weights.

        Args:
          file_path: Optional checkpoint path.
          cfg: Optional model config.
          state_dict: Optional model weights.

        Returns:
          Instantiated model.
        """
        cfg, state_dict = HyperTorchModel._load_cfg_state_dict(
            file_path, cfg, state_dict
        )

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def get_reg_loss(self) -> int:
        """Return regularization loss contribution for this model.

        Returns:
          Regularization loss term.
        """
        return 0

    def get_loss(self) -> int:
        """Return auxiliary loss contribution for this model.

        Returns:
          Auxiliary loss term.
        """
        return 0

    @property
    def device(self) -> torch.device:
        """Return unique device shared by parameters and buffers.

        Returns:
          Device where all parameters and buffers reside.
        """
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
    def _remove_module_prefix(
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remove leading ``module.`` prefixes from state-dict keys.

        Args:
          state_dict: State dict to normalize.

        Returns:
          Normalized state dict.
        """
        import re

        if not state_dict:
            return state_dict

        p = re.compile(r"^(module\.)+")
        if p.match(list(state_dict.keys())[0]) is not None:
            state_dict = ODict((p.sub("", k), v) for k, v in state_dict.items())

        return state_dict

    @staticmethod
    def _fix_xvector_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize legacy XVector config keys to current names.

        Args:
          cfg: Model config dictionary.

        Returns:
          Updated config dictionary.
        """
        # We renamed AM-softmax scale parameer s to cos_scale
        if "s" in cfg:
            cfg["cos_scale"] = cfg.pop("s")

        return cfg

    @staticmethod
    def _fix_hf_wav2xvector(
        cfg: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Migrate legacy HF wav2xvector fusion config and checkpoint layout.

        Args:
          cfg: Model config dictionary.
          state_dict: Model checkpoint weights.

        Returns:
          Updated ``(cfg, state_dict)``.
        """
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
    def _fix_resnet_qvector_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Drop deprecated ResNetQVector config keys.

        Args:
          cfg: Model config dictionary.

        Returns:
          Updated config dictionary.
        """
        if "proj_head" in cfg:
            del cfg["proj_head"]

        return cfg

    @staticmethod
    def _fix_model_compatibility(
        class_obj: Type["HyperTorchModel"],
        cfg: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Apply compatibility fixes for deprecated model formats.

        Args:
          class_obj: Model class to instantiate.
          cfg: Configuration dictionary.
          state_dict: Serialized model weights.

        Returns:
          Updated ``(cfg, state_dict)``.
        """
        # for compatibility with older x-vector models
        XVector = HyperTorchModel.registry.get("XVector", None)
        if XVector is not None and issubclass(class_obj, XVector):
            cfg = HyperTorchModel._fix_xvector_cfg(cfg)

        # switch old feature fuser to new feature fuser in w2v x-vectors
        HFWav2XVector = HyperTorchModel.registry.get("HFWav2XVector", None)
        if HFWav2XVector is not None and issubclass(class_obj, HFWav2XVector):
            cfg, state_dict = HyperTorchModel._fix_hf_wav2xvector(cfg, state_dict)

        # switch audio_feats params to buffers
        Wav2XVector = HyperTorchModel.registry.get("Wav2XVector", None)
        if Wav2XVector is not None and issubclass(class_obj, Wav2XVector):
            # Remove _window if it was saved as a parameter
            for key in list(state_dict.keys()):
                if key.endswith("_window"):
                    tensor = state_dict.pop(key)
                    # keep same key, PyTorch will now map it as a buffer
                    state_dict[key] = tensor
                if key.endswith("_fb"):
                    tensor = state_dict.pop(key)
                    # keep same key, PyTorch will now map it as a buffer
                    state_dict[key] = tensor

        # Remove QVector bugs in first implementation
        ResNetQVector = HyperTorchModel.registry.get("ResNetQVector", None)
        if ResNetQVector is not None and issubclass(class_obj, ResNetQVector):
            cfg = HyperTorchModel._fix_resnet_qvector_cfg(cfg)

        return cfg, state_dict

    @staticmethod
    def _is_hf_path(file_path: Path) -> bool:
        """Return ``True`` for HF-style ``org/repo/file`` paths.

        Args:
          file_path: Path to validate.

        Returns:
          ``True`` if path shape matches expected HF format.
        """
        parts = file_path.parts
        if file_path.is_absolute():
            parts = parts[1:]

        # Expected format: org_or_user/repo/filename
        return len(parts) == 3 and all(part != "" for part in parts)

    @staticmethod
    def _get_from_hf(
        file_path: Path, cache_dir: PathLike = None, local_dir: PathLike = None
    ) -> str:
        """Download a file from Hugging Face Hub and return its local path.

        Args:
          file_path: HF-style path ``org/repo/file``.
          cache_dir: Optional HF cache directory.
          local_dir: Optional local download directory.

        Returns:
          Local filesystem path to downloaded file.
        """
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
    ) -> Path:
        """Resolve local file path, downloading from HF Hub when needed.

        Args:
          file_path: Local path or prefixed HF path.
          cache_dir: Optional HF cache directory.
          local_dir: Optional local download directory.

        Returns:
          Resolved local path.
        """
        if str(file_path)[:3] == "hf:":
            # hf: prefix indicates to download from hub
            file_path = Path(str(file_path)[3:])
            assert HyperTorchModel._is_hf_path(
                file_path
            ), f"{file_path} is not a valid HF path"
            file_path = HyperTorchModel._get_from_hf(
                file_path, cache_dir=cache_dir, local_dir=local_dir
            )
            return Path(file_path)
        elif not file_path.is_file():
            # if no prefix but file not in local dir try to get it from hub
            if not HyperTorchModel._is_hf_path(file_path):
                return file_path

            try:
                file_path = HyperTorchModel._get_from_hf(file_path)
                return Path(file_path)
            except Exception as err:
                logging.debug("failed to fetch checkpoint from HF hub: %s", err)
                return file_path

        else:
            # file is local
            return file_path

    @staticmethod
    def _bootstrap_registry() -> None:
        """Import common torch subpackages so subclasses register themselves.

        ``HyperTorchModel.registry`` is populated when subclass definitions are
        imported. CLI scripts often import only ``HyperTorchModel`` and call
        ``auto_load``, so we need a lazy import step before class lookup.
        """
        # Import packages that actually define HyperTorchModel subclasses.
        # We ignore import errors here to preserve backward compatibility in
        # environments missing optional dependencies.
        module_names = (
            "hyperion.torch.narchs",
            "hyperion.torch.models",
            "hyperion.torch.tpm",
        )
        for module_name in module_names:
            try:
                import_module(module_name)
            except Exception as err:
                logging.debug(
                    "Skipping registry bootstrap import %s: %s", module_name, err
                )

    @staticmethod
    def _find_module_for_class_name(class_name: str) -> Optional[str]:
        """Find module path for class name by scanning torch source files.

        Args:
          class_name: Target class name.

        Returns:
          Dotted module path if found, otherwise ``None``.
        """
        torch_dir = Path(__file__).resolve().parent
        search_dirs = (
            torch_dir / "models",
            torch_dir / "narchs",
            torch_dir / "layers",
            torch_dir / "layer_blocks",
            torch_dir / "tpm",
        )
        class_decl = f"class {class_name}("
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue

            for py_file in search_dir.rglob("*.py"):
                try:
                    text = py_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                if class_decl not in text:
                    continue

                module_path = (
                    Path("hyperion")
                    / "torch"
                    / py_file.relative_to(torch_dir).with_suffix("")
                )
                return ".".join(module_path.parts)

        return None

    @staticmethod
    def auto_load(
        file_path: PathLike,
        model_name: Optional[str] = None,
        extra_objs: Optional[Dict[str, Type["HyperTorchModel"]]] = None,
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
    ) -> "HyperTorchModel":
        """Load model from checkpoint with dynamic class resolution.

        Args:
          file_path: Local path or HF-style path.
          model_name: State-dict prefix; defaults to ``model``.
          extra_objs: Optional mapping from class names to model classes.
          map_location: ``torch.load`` map location.
          cache_dir: Optional HF cache directory.
          local_dir: Optional local download directory.

        Returns:
          Loaded model instance.
        """
        if extra_objs is None:
            extra_objs = {}

        file_path = Path(file_path)
        file_path = HyperTorchModel._try_to_get_from_hf(
            file_path, cache_dir=cache_dir, local_dir=local_dir
        )

        assert file_path.is_file(), f"HyperTorchModel file: {file_path} not found"

        if map_location is None:
            map_location = torch.device("cpu")

        model_data = torch.load(
            file_path,
            map_location=map_location,
            # PyTorch 2.6 default weights_only=True breaks config objects
            # serialized in model_cfg (e.g., custom enums/types).
            weights_only=False,
        )
        cfg = model_data["model_cfg"]

        if "class_name" not in cfg:
            cfg["class_name"] = "DAC"

        class_name = cfg["class_name"]
        del cfg["class_name"]
        if class_name not in HyperTorchModel.registry:
            HyperTorchModel._bootstrap_registry()
        if class_name not in HyperTorchModel.registry:
            module_name = HyperTorchModel._find_module_for_class_name(class_name)
            if module_name is not None:
                try:
                    import_module(module_name)
                except Exception as err:
                    raise Exception(
                        "failed to import module %s for class_name=%s (%s)"
                        % (module_name, class_name, err)
                    ) from err

        if class_name in HyperTorchModel.registry:
            class_obj = HyperTorchModel.registry[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception("unknown object with class_name=%s" % (class_name))

        if model_name is None:
            model_name = "model"
        state_dict = model_data[f"{model_name}_state_dict"]

        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        state_dict = HyperTorchModel._remove_module_prefix(state_dict)
        cfg, state_dict = HyperTorchModel._fix_model_compatibility(
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


class TorchModel(HyperTorchModel):
    """Backward-compatible alias for HyperTorchModel."""

    pass
