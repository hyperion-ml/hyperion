"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import logging
import pickle
import numpy as np
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo

from sklearn.svm import SVC as SVC

from ...hyp_defs import float_cpu
from ..np_model import NPModel
from ...utils.math import softmax


class GaussianSVMC(NPModel):
    """Gaussian Support Vector Machine for Classification."""

    def __init__(
        self,
        C=1.0,
        gamma="scale",
        shrinking=True,
        probability=True,
        tol=0.0001,
        cache_size=600,
        multi_class="ovr",
        break_ties=True,
        class_weight=None,
        random_state=None,
        max_iter=100,
        model=None,
        verbose=0,
        balance_class_weight=True,
        lr_seed=1024,
        labels=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        if class_weight is None and balance_class_weight:
            class_weight = "balanced"

        if random_state is None:
            random_state = np.random.RandomState(seed=lr_seed)

        self.balance_class_weight = balance_class_weight
        if model is None:
            self.svm = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                shrinking=shrinking,
                probability=probability,
                tol=tol,
                cache_size=cache_size,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
                decision_function_shape=multi_class,
                break_ties=break_ties,
                random_state=random_state,
            )
        else:
            self.svm = model
        self.set_labels(labels)

    @property
    def model_params(self):
        return self.svm.get_params()

    def set_labels(self, labels):
        if isinstance(labels, np.ndarray):
            labels = list(labels)
        self.labels = labels

    def get_config(self):
        """Gets configuration hyperparams.
        Returns:
          Dictionary with config hyperparams.
        """
        config = {
            "balance_class_weight": self.balance_class_weight,
            "labels": self.labels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def predict(self, x, eval_type="decision-func"):
        """Evaluates the SVM

        Args:
          x: input features (num_samples, feat_dim),
             it can be (num_samples,) if feat_dim=1.
          eval_type: evaluationg method: logit (log-likelihood ratio),
                     bin-log-post (binary log-posteriors),
                     bin-post (binary posteriors)
                     cat-log-post (categorical log-posteriors),
                     cat-post (categorical posteriors)
        Returns:
          Ouput scores (num_samples, num_classes)
        """
        if eval_type == "cat-post":
            return self.svm.predict_proba(x)
        if eval_type == "cat-log-post":
            return self.svm.predict_log_proba(x)

        return self.svm.decision_function(x)

    def __call__(self, x, eval_type="decision-func"):
        """Evaluates the SVM

        Args:
          x: input features (num_samples, feat_dim),
             it can be (num_samples,) if feat_dim=1.
          eval_type: evaluationg method: logit (log-likelihood ratio),
                     bin-log-post (binary log-posteriors),
                     bin-post (binary posteriors)
                     cat-log-post (categorical log-posteriors),
                     cat-post (categorical posteriors)
        Returns:
          Ouput scores (num_samples, num_classes)
        """
        return self.predict(x, eval_type)

    def fit(self, x, class_ids, sample_weight=None):
        """Estimates the parameters of the model.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          class_ids: class integer [0, num_classes-1] identifier (num_samples,)
          sample_weight: weight of each sample in the estimation (num_samples,)
        """
        print("--------------", type(x[3, 2]), type(class_ids[20]), "--------------")
        self.svm.fit(x, class_ids)
        if self.svm.fit_status_:
            logging.warning("SVM did not converge")

    def save(self, file_path):
        """Saves the model to file.

        Args:
          file_path: filename path.
        """
        file_dir = os.path.dirname(file_path)
        if not (os.path.isdir(file_dir)):
            os.makedirs(file_dir, exist_ok=True)
        split_path = os.path.splitext(file_path)
        if not split_path[-1] == "sav":
            file_path = "".join(split_path[0] + ".sav")
        with open(file_path, "wb") as f:
            # with h5py.File(file_path, "w") as f:
            # config = self.to_json()
            # f.create_dataset("config", data=np.array(config, dtype="S"))
            self.save_params(f)

    @classmethod
    def load(cls, file_path):
        """Loads the model from file.

        Args:
          file_path: path to the file where the model is stored.

        Returns:
          Model object.
        """
        split_path = os.path.splitext(file_path)
        if not split_path[-1] == "sav":
            file_path = "".join(split_path[0] + ".sav")

        # with h5py.File(file_path, "r") as f:
        with open(file_path, "rb") as f:
            # json_str = str(np.asarray(f["config"]).astype("U"))
            # config = cls.load_config_from_json(json_str)
            config = None
            return cls.load_params(f, config)

    def save_params(self, f):
        # params = {"A": self.A, "b": self.b}
        # self._save_params_from_dict(f, params)
        pickle.dump(self, f)

    @classmethod
    def load_params(cls, f, config):
        # param_list = ["A", "b"]
        # params = cls._load_params_to_dict(f, config["name"], param_list)
        # kwargs = dict(list(config.items()) + list(params.items()))
        # return cls(**kwargs)
        svmc = pickle.load(f)
        return svmc

    @staticmethod
    def filter_class_args(**kwargs):
        """Extracts the hyperparams of the class from a dictionary.

        Returns:
          Hyperparamter dictionary to initialize the class.
        """
        valid_args = (
            "nu",
            "gamma",
            "shrinking",
            "probability",
            "tol",
            "cache_size",
            "multi_class",
            "break_ties",
            "class_weight",
            "random_state",
            "max_iter",
            "verbose",
            "balance_class_weight",
            "lr_seed",
            "model",
            "labels",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    filter_train_args = filter_class_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        """It adds the arguments corresponding to the class to jsonarparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--c",
            dest="C",
            default=1.0,
            type=float,
            help="inverse of regularization strength",
        )
        # parser.add_argument(
        #     "--class_weight",
        #     default=None,
        #     help="Class weights",
        # )
        parser.add_argument(
            "--gamma",
            default="scale",
            choices=["scale", "auto"],
            help="Kernel coefficient for ‘rbf’",
        )
        parser.add_argument(
            "--shrinking",
            default=True,
            type=bool,
            help="Whether to use the shrinking heuristic",
        )
        parser.add_argument(
            "--probability",
            default=True,
            type=bool,
            help="Whether to enable probability estimates",
        )
        parser.add_argument(
            "--break_ties",
            default=True,
            type=bool,
            help="If true, predict will break ties according to the confidence values of decision_function; otherwise \
            the first class among the tied classes is returned",
        )
        parser.add_argument(
            "--lr-seed", default=1024, type=int, help="random number generator seed"
        )
        parser.add_argument(
            "--max-iter",
            dest="max_iter",
            default=100,
            type=int,
            help="only for the newton-cg, sag and lbfgs solvers",
        )
        parser.add_argument(
            "--tol", default=1e-4, type=float, help="tolerance for stopping criteria"
        )
        parser.add_argument(
            "--multi-class",
            default="ovr",
            choices=["ovr", "ovo"],
            help=(
                "ovr fits a binary problem for each class else "
                "it minimizes the multinomial loss."
            ),
        )
        parser.add_argument(
            "--cache_size",
            default=600,
            type=int,
            help="Specify the size of the kernel cache (in MB)",
        )
        parser.add_argument(
            "--verbose",
            default=0,
            type=int,
            help="For the liblinear and lbfgs solvers",
        )
        parser.add_argument(
            "--balance-class-weight",
            default=False,
            action=ActionYesNo,
            help="Balances the weight of each class when computing W",
        )
        parser.add_argument("--name", default="svc", help="model name")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix, action=ActionParser(parser=parser),
            )

    @staticmethod
    def filter_eval_args(**kwargs):
        """Extracts the evaluation time hyperparams of the class from a dictionary.

        Returns:
          Hyperparameters to evaluate the class.
        """
        valid_args = "eval_type"
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser, prefix=None):
        """It adds the arguments needed to evaluate the class to jsonarparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--eval-type",
            default="decision-func",
            choices=["cat-log-post", "cat-post", "decision-func"],
            help=("type of evaluation"),
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix, action=ActionParser(parser=parser),
            )

    # for backward compatibility
    filter_train_args = filter_class_args
    add_argparse_args = add_class_args
    add_argparse_train_args = add_class_args
    add_argparse_eval_args = add_eval_args
