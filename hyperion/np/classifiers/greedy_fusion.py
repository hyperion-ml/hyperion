"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np

from ...hyp_defs import float_cpu, float_save
from ..np_model import NPModel
from ..metrics import dcf

from .binary_logistic_regression import BinaryLogisticRegression as BLR


class GreedyFusionBinaryLR(NPModel):
    """Greedy score fusion based on binary logistic regression.

    It computes ``max_systmes`` fusions. The best system, the best fusion of two,
    the best fusion of three, ...
    The system selection procedure is as follows:
    * Choose the best system.
    * Fix the best system and choose the system that fuses the best with the best.
    * Fix the best two and choose the system that fuses the best with those two.
    * ...

    Attributes:
      weights: fusion weights, this is a list with ``max_systems`` elements with shapes, (1,1), (2,1), (3,1), ..., (max_systems,1).
      bias: fusion biaes, this is a list with ``max_systems`` elements with shape (1,).
      system_idx: list of index vector that indicate, which systems are used for the fusion of 1 system, fusion of 2, ....
      system_names: list of strings containing descriptive names for the systems,
      max_systems: max number of systems to fuse, if None, ``max_systems=total_systems``.
      penalty: str, ‘l1’ or ‘l2’, default: ‘l2’ ,
                 Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
                  New in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)
      lambda_reg: float, default: 1e-5
                     Regularization strength; must be a positive float.
      use_bias: bool, default: True
                   Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
      bias_scaling: float, default 1.
                       Useful only when the solver ‘liblinear’ is used and use_bias is set to True.
                       In this case, x becomes [x, bias_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.
                       Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) bias_scaling has to be increased.
      priors: prior prob for having a positive sample.
      random_state: int, RandomState instance or None, optional, default: None
                       The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; . Used when solver == ‘sag’ or ‘liblinear’.
      solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
                 default: ‘liblinear’ Algorithm to use in the optimization problem.
                 For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
                 ‘saga’ are faster for large ones.
                 ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas
                 ‘liblinear’ and ‘saga’ handle L1 penalty.
                 Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.
      max_iter: int, default: 100
                   Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
      dual: bool, default: False
               Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
      tol: float, default: 1e-4
              Tolerance for stopping criteria.
      verbose: int, default: 0
                  For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
      warm_start: bool, default: False
                     When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.
                     New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers.
      lr_seed: seed for numpy random.

    """

    def __init__(
        self,
        weights=None,
        bias=None,
        system_idx=None,
        system_names=None,
        max_systems=None,
        prioritize_positive=True,
        penalty="l2",
        lambda_reg=1e-6,
        bias_scaling=1,
        prior=0.5,
        prior_eval=None,
        solver="liblinear",
        max_iter=100,
        dual=False,
        tol=0.0001,
        verbose=0,
        lr_seed=1024,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.weights = weights
        self.bias = bias
        self.system_idx = system_idx
        self.system_names = system_names
        self.max_systems = max_systems
        self.prioritize_positive = prioritize_positive
        if prior_eval is None:
            self.prior_eval = prior
        else:
            self.prior_eval = prior_eval

        self.lr = BLR(
            penalty=penalty,
            lambda_reg=lambda_reg,
            use_bias=True,
            bias_scaling=bias_scaling,
            prior=prior,
            solver=solver,
            max_iter=max_iter,
            dual=dual,
            tol=tol,
            verbose=verbose,
            warm_start=False,
            lr_seed=lr_seed,
        )

    @property
    def prior(self):
        """Prior probability for a positive sample."""
        return self.lr.prior

    def get_fusion_params(self, idx):
        """Get fusion parameters for a fusion of ``idx+1`` systems.

        Args:
          idx: index of the fusion, it returns the parameters for the fusion of ``idx+1`` systems.
        Returns:
          Weights for fusion ``idx`` shape=(idx+1, 1)
          Bias for fusion ``idx``
          Indices for systems incuded in fusion ``idx``.
        """
        return self.weights[idx], self.bias[idx], self.system_idx[idx]

    def _predict_fus_idx(self, x, fus_idx, eval_type="logit"):
        """Evals the fusion indicated by ``fus_idx``,
        which is the fusion of ``fus_idx+1`` systems.

        Args:
          x: input features (num_samples, num_systems)
          fus_idx: index of the fusion, it returns the parameters for the fusion of ``fus_idx+1`` systems.
          eval_type: evaluationg method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Ouput scores (num_samples,)
        """

        w, b, idx = self.get_fusion_params(fus_idx)
        x = x[:, idx]
        y = np.dot(x, w).ravel() + b

        if eval_type == "log-post":
            y = np.log(softmax(y + np.log(self.priors), axis=1) + 1e-10)
        if eval_type == "post":
            y = softmax(y + np.log(self.priors))

        return y

    def predict(self, x, fus_idx=None, eval_type="logit"):
        """Evals the fusion indicated by ``fus_idx``,
        which is the fusion of ``fus_idx+1`` systems.

        Args:
          x: input features (num_samples, num_systems)
          fus_idx: index of the fusion, it returns the parameters for the fusion of ``fus_idx+1`` systems.
                   If None, it evals all the fusions and return a list of score vectors
          eval_type: evaluationg method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Ouput scores (num_samples,) or List of score vectors.
        """

        if fus_idx is None:
            y = []
            for i in range(len(self.weights)):
                y_i = self._predict_fus_idx(x, i, eval_type)
                y.append(y_i)
            return y

        return self._predict_fus_idx(x, fus_idx, eval_type)

    def __call__(self, x, fus_idx=None, eval_type="logit"):
        """Evals the fusion indicated by ``fus_idx``,
        which is the fusion of ``fus_idx+1`` systems.

        Args:
          x: input features (num_samples, num_systems)
          fus_idx: index of the fusion, it returns the parameters for the fusion of ``fus_idx+1`` systems.
                   If None, it evals all the fusions and return a list of score vectors
          eval_type: evaluationg method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Ouput scores (num_samples,) or List of score vectors.
        """
        return self.predict(x, fus_idx, eval_type)

    def fit(self, x, class_ids, sample_weights=None):
        """Estimates the parameters of all the fusions

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          class_ids: class integer [0, 1] identifier (num_samples,)
          sample_weight: weight of each sample in the estimation (num_samples,)
        """

        num_systems = x.shape[1]
        if self.max_systems is None:
            self.max_systems = 10

        self.max_systems = min(self.max_systems, num_systems)

        self.weights = []
        self.bias = []
        self.system_idx = []
        fus_min_dcf = np.zeros((self.max_systems,), dtype=float_cpu())
        fus_act_dcf = np.zeros((self.max_systems,), dtype=float_cpu())
        for i in range(self.max_systems):
            cand_systems = np.arange(num_systems, dtype="int32")
            fixed_systems = np.array([], dtype="int32")
            if i > 0:
                fixed_systems = self.system_idx[i - 1]
                cand_systems[fixed_systems] = -1
                cand_systems = cand_systems[cand_systems > -1]

            num_cands = len(cand_systems)
            cand_min_dcf = np.zeros((num_cands,), dtype=float_cpu())
            cand_act_dcf = np.zeros((num_cands,), dtype=float_cpu())
            all_pos = np.zeros((num_cands,), dtype=np.bool)
            cand_weights = []
            for j in range(num_cands):
                system_idx_ij = np.concatenate(
                    (fixed_systems, np.expand_dims(cand_systems[j], axis=0)), axis=0
                )
                x_ij = x[:, system_idx_ij]
                self.lr.fit(x_ij, class_ids)
                cand_weights.append([self.lr.A, self.lr.b])
                all_pos[j] = np.all(self.lr.A > 0)

                y_ij = self.lr.predict(x_ij)
                tar = y_ij[class_ids == 1]
                non = y_ij[class_ids == 0]
                min_dcf, act_dcf, _, _ = dcf.fast_eval_dcf_eer(
                    tar, non, self.prior_eval
                )
                cand_min_dcf[j] = np.mean(min_dcf)
                cand_act_dcf[j] = np.mean(act_dcf)

                fus_name = self._make_fus_name(system_idx_ij)
                logging.info(
                    "fus_sys=%s min_dcf=%.3f act_dcf=%.3f"
                    % (fus_name, cand_min_dcf[j], cand_act_dcf[j])
                )

            dcf_best = 100
            if self.prioritize_positive:
                allpos_cand_act_dcf = np.copy(cand_act_dcf)
                allpos_cand_act_dcf[all_pos == False] = 100
                j_best = np.argmin(allpos_cand_act_dcf)
                dcf_best = allpos_cand_act_dcf[j_best]

            if dcf_best == 100:
                j_best = np.argmin(cand_act_dcf)
                dcf_best = cand_act_dcf[j_best]

            select_system = np.asarray([cand_systems[j_best]])
            if i == 0:
                fus_system_i = select_system
            else:
                fus_system_i = np.concatenate((self.system_idx[i - 1], select_system))

            self.system_idx.append(fus_system_i)

            weights_i, bias_i = cand_weights[j_best]
            self.weights.append(weights_i)
            self.bias.append(bias_i)
            fus_min_dcf[i] = cand_min_dcf[j_best]
            fus_act_dcf[i] = cand_act_dcf[j_best]

        # print report
        for i in range(self.max_systems):
            fus_name = self._make_fus_name(self.system_idx[i])
            weights_str = (
                np.array2string(self.weights[i].ravel(), separator=",")
                .replace("\r", "")
                .replace("\n", "")
            )
            bias_str = np.array2string(self.bias[i], separator=",")
            logging.info(
                "Best-%d=%s min_dcf=%.3f act_dcf=%.3f weights=%s bias=%s"
                % (
                    i + 1,
                    fus_name,
                    fus_min_dcf[i],
                    fus_act_dcf[i],
                    weights_str,
                    bias_str,
                )
            )

        return fus_min_dcf, fus_act_dcf

    def _make_fus_name(self, idx):
        sys_names = [self.system_names[i] for i in idx]
        fus_name = "+".join(sys_names)
        return fus_name

    def get_config(self):
        """Gets configuration hyperparams.
        Returns:
          Dictionary with config hyperparams.
        """
        config = {"bias_scaling": self.lr.bias_scaling, "prior": self.lr.prior}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        weights = np.concatenate(tuple(self.weights), axis=0)
        bias = np.concatenate(tuple(self.bias))
        system_idx = np.concatenate(tuple(self.system_idx), axis=0)
        system_names = np.asarray(self.system_names, dtype="S")
        params = {
            "weights": weights,
            "bias": bias,
            "system_idx": system_idx,
            "system_names": system_names,
        }
        dtypes = {
            "weights": float_save(),
            "bias": float_save(),
            "system_idx": "int32",
            "system_names": "S",
        }

        self._save_params_from_dict(f, params, dtypes=dtypes)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["weights", "bias", "system_idx", "system_names"]
        dtypes = {
            "weights": float_cpu(),
            "bias": float_cpu(),
            "system_idx": "int32",
            "system_names": "S",
        }
        params = cls._load_params_to_dict(f, config["name"], param_list, dtypes)

        weights = []
        system_idx = []
        i = 1
        j = 0
        while j < params["weights"].shape[0]:
            weights.append(params["weights"][j : j + i, :])
            system_idx.append(params["system_idx"][j : j + i])
            j += i
            i += 1

        params["weights"] = weights
        params["system_idx"] = system_idx
        params["system_names"] = [t.decode("utf-8") for t in params["system_names"]]

        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)
