"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import json
import logging
import pickle
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Type, Union

import h5py
import numpy as np

from ..hyp_defs import float_cpu, float_save
from ..utils.misc import PathLike


class HyperNPModel:
    """Base class for machine learning models based on numpy.

    Attributes:
      name: optional identifier for the model.
    """

    registry: ClassVar[Dict[str, Type["HyperNPModel"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        HyperNPModel.registry[cls.__name__] = cls

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize base model metadata.

        Args:
          name: Optional identifier for the model instance. If ``None``, the
            class name is used.
          **kwargs: Reserved for subclass compatibility.
        """
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self._is_init = False

    def copy(self) -> "HyperNPModel":
        """Returns a clone of the model."""
        return deepcopy(self)

    def clone(self) -> "HyperNPModel":
        """Returns a clone of the model."""
        return deepcopy(self)

    @property
    def is_init(self) -> bool:
        """Returns True if the model has been initialized."""
        return self._is_init

    def init_to_false(self) -> None:
        """Sets the model as non initialized."""
        self._is_init = False

    def initialize(self) -> None:
        """Initialize model parameters/state.

        Subclasses can override this method when they have lazy initialization
        logic.
        """
        pass

    def fit(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the model.

        Args:
          x: train data matrix with shape (num_samples, x_dim).
          sample_weight: weight of each sample in the training loss shape (num_samples,).
          x_val: validation data matrix with shape (num_val_samples, x_dim).
          sample_weight_val: weight of each sample in the val. loss.

        Raises:
          NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError()

    def fit_generator(self, x: Any, x_val: Optional[Any] = None) -> None:
        """Trains the model from a data generator function.

        Args:
          x: train data generation function.
          x_val: validation data generation function.

        Raises:
          NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError()

    def save(self, file_path: PathLike) -> None:
        """Saves the model to file.

        Args:
          file_path: filename path.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix.lower() == ".pkl":
            with file_path.open("wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        with h5py.File(file_path, "w") as f:
            config = self.to_json()
            f.create_dataset("config", data=np.array(config, dtype="S"))
            self.save_params(f)

    def save_params(self, f: h5py.File) -> None:
        """Saves the model paramters into the file.

        Args:
          f: file handle.

        Raises:
          NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError(
            f"save_params method not defined for {self.__class__.__name__}"
        )

    def _save_params_from_dict(
        self,
        f: h5py.File,
        params: Mapping[str, Any],
        dtypes: Optional[Union[type, Mapping[str, Any]]] = None,
    ) -> None:
        """Saves a dictionary of model parameters into the file.

        Args:
          f: file handle.
          params: dictionary of model parameters.
          dtypes: dictionary indicating the dtypes of the model parameters.
        """
        if dtypes is None:
            dtypes = dict((k, float_save()) for k in params)
        elif isinstance(dtypes, type):
            dtypes = dict((k, dtypes) for k in params)

        if self.name is None:
            prefix = ""
        else:
            prefix = self.name + "/"
        for k, v in params.items():
            if v is None:
                continue
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            p_name = prefix + k
            f.create_dataset(p_name, data=v.astype(dtypes[k], copy=False))

    @classmethod
    def load_config(cls, file_path: PathLike) -> Dict[str, Any]:
        """Loads the model configuration from file.

        Args:
          file_path: path to the file where the model is stored.

        Returns:
          Dictionary containing the model configuration.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pkl":
            with file_path.open("rb") as f:
                model = pickle.load(f)
            if not isinstance(model, HyperNPModel):
                raise TypeError(
                    f"Expected HyperNPModel in pickle, got {type(model).__name__}"
                )
            return model.get_config()

        try:
            with h5py.File(file_path, "r") as f:
                json_str = str(np.asarray(f["config"]).astype("U"))
                return cls.load_config_from_json(json_str)
        except OSError:
            with file_path.open("r", encoding="utf-8") as f:
                return cls.load_config_from_json(f.read())

    @classmethod
    def load(cls, file_path: PathLike) -> "HyperNPModel":
        """Loads the model from file.

        Args:
          file_path: path to the file where the model is stored.

        Returns:
          Model object.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pkl":
            with file_path.open("rb") as f:
                model = pickle.load(f)
            if not isinstance(model, cls):
                raise TypeError(
                    f"Expected {cls.__name__} in pickle, got {type(model).__name__}"
                )
            return model

        with h5py.File(file_path, "r") as f:
            json_str = str(np.asarray(f["config"]).astype("U"))
            config = cls.load_config_from_json(json_str)
            return cls.load_params(f, config)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "HyperNPModel":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        return cls(name=config["name"])

    @staticmethod
    def _load_params_to_dict(
        f: h5py.File,
        name: Optional[str],
        params: Sequence[str],
        dtypes: Optional[Union[type, Mapping[str, Any]]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """Loads the model parameters from file to a dictionary.

        Args:
          f: file handle.
          name: model identifier or None.
          params: parameter names.
          dtypes: dictionary containing the dtypes of the parameters.

        Returns:
          Dictionary with model parameters.
        """
        if dtypes is None:
            dtypes = dict((k, float_cpu()) for k in params)
        elif isinstance(dtypes, type):
            dtypes = dict((k, dtypes) for k in params)

        if name is None:
            prefix = ""
        else:
            prefix = name + "/"

        param_dict = {}
        for k in params:
            p_name = prefix + k
            if p_name in f:
                param_dict[k] = np.asarray(f[p_name]).astype(
                    dtype=dtypes[k], copy=False
                )
            else:
                param_dict[k] = None
        return param_dict

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {"class_name": self.__class__.__name__, "name": self.name}
        return config

    def to_json(self, **kwargs: Any) -> str:
        """Return model configuration serialized as JSON.

        Args:
          **kwargs: Extra keyword arguments forwarded to :func:`json.dumps`.

        Returns:
          JSON string with model configuration.
        """

        def get_json_type(obj: Any) -> Any:
            # if obj is a np list of strings
            if isinstance(obj, np.ndarray) and obj.ndim == 1:
                if isinstance(obj[0], str):
                    return list(obj)

            # Piece of code borrowed from keras
            # if obj is any numpy type
            if type(obj).__module__ == np.__name__:
                return obj.item()

            # if obj is a python 'type'
            if type(obj).__name__ == type.__name__:
                return obj.__name__

            raise TypeError("Not JSON Serializable:", obj)

        config = self.get_config()
        return json.dumps(config, default=get_json_type, **kwargs)

    @staticmethod
    def load_config_from_json(json_str: str) -> Dict[str, Any]:
        """Convert JSON configuration string to dictionary."""
        return json.loads(json_str)

    @staticmethod
    def _bootstrap_registry() -> None:
        """Import common NP subpackages so subclasses register themselves."""
        module_names = (
            "hyperion.np.pdfs",
            "hyperion.np.transforms",
            "hyperion.np.classifiers",
            "hyperion.np.calibration",
            "hyperion.np.clustering",
            "hyperion.np.score_norm",
        )
        for module_name in module_names:
            try:
                import_module(module_name)
            except Exception as err:
                logging.debug(
                    "Skipping NP registry bootstrap import %s: %s", module_name, err
                )

    @staticmethod
    def _find_module_for_class_name(class_name: str) -> Optional[str]:
        """Find module path for a registered class name by scanning NP sources.

        Args:
          class_name: Target class name to locate.

        Returns:
          Dotted module path if found, otherwise ``None``.
        """
        np_dir = Path(__file__).resolve().parent
        search_dirs = (
            np_dir / "pdfs",
            np_dir / "transforms",
            np_dir / "classifiers",
            np_dir / "calibration",
            np_dir / "clustering",
            np_dir / "score_norm",
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
                    / "np"
                    / py_file.relative_to(np_dir).with_suffix("")
                )
                return ".".join(module_path.parts)

        return None

    @staticmethod
    def auto_load(
        file_path: PathLike,
        extra_objs: Optional[Dict[str, Type["HyperNPModel"]]] = None,
    ) -> "HyperNPModel":
        """Auto-load a serialized model based on the saved ``class_name``.

        Args:
          file_path: Path to model file.
          extra_objs: Optional mapping from class name to class object used as a
            fallback when class is not yet registered.

        Returns:
          Instantiated model loaded from ``file_path``.

        Raises:
          Exception: If the class cannot be resolved/imported.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pkl":
            with file_path.open("rb") as f:
                model = pickle.load(f)
            if not isinstance(model, HyperNPModel):
                raise TypeError(
                    f"Expected HyperNPModel in pickle, got {type(model).__name__}"
                )
            return model

        if extra_objs is None:
            extra_objs = {}

        class_name = HyperNPModel.load_config(file_path)["class_name"]
        if class_name not in HyperNPModel.registry:
            HyperNPModel._bootstrap_registry()
        if class_name not in HyperNPModel.registry:
            module_name = HyperNPModel._find_module_for_class_name(class_name)
            if module_name is not None:
                try:
                    import_module(module_name)
                except Exception as err:
                    raise Exception(
                        "failed to import module %s for class_name=%s (%s)"
                        % (module_name, class_name, err)
                    ) from err

        if class_name in HyperNPModel.registry:
            class_obj = HyperNPModel.registry[class_name]
        elif class_name in extra_objs:
            class_obj = extra_objs[class_name]
        else:
            raise Exception("unknown object with class_name=%s" % (class_name))

        return class_obj.load(file_path)


class NPModel(HyperNPModel):
    """Backward-compatible alias for HyperNPModel."""

    pass
