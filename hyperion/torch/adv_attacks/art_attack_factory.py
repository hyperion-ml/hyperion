"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

try:
    from art.attacks import evasion as attacks
except ImportError:
    pass

from ...utils.misc import filter_func_args


def make_4d_hook(func):
    def wrapper(x, *args, **kwargs):
        x = x[None, None]
        y = func(x, *args, **kwargs)
        return y[0, 0]

    return wrapper


class ARTAttackFactory(object):
    @staticmethod
    def create(
        model,
        attack_type,
        eps=0,
        delta=0.01,
        step_adapt=0.667,
        num_trial=25,
        sample_size=20,
        init_size=100,
        norm=np.inf,
        eps_step=0.1,
        num_random_init=0,
        minimal=False,
        random_eps=False,
        min_eps=1e-6,
        beta=0.001,
        theta=0.1,
        gamma=1.0,
        confidence=0.0,
        lr=1e-2,
        lr_decay=0.5,
        lr_num_decay=20,
        momentum=0.8,
        binary_search_steps=9,
        max_iter=10,
        overshoot=1.1,
        num_grads=10,
        max_halving=5,
        max_doubling=5,
        tau_decr_factor=0.9,
        initial_c=1e-5,
        largest_c=20.0,
        c_incr_factor=2.0,
        decision_rule="EN",
        init_eval=100,
        max_eval=10000,
        num_parallel=128,
        variable_h=1e-4,
        use_importance=False,
        abort_early=True,
        th=None,
        es: int = 0,
        sigma=0.5,
        lambda_tv=0.3,
        lambda_c=1.0,
        lambda_s=0.5,
        reg=3000,
        kernel_size=5,
        eps_factor=1.1,
        eps_iter=10,
        p_wassertein=2,
        conj_sinkhorn_iter=400,
        proj_sinkhorn_iter=400,
        sub_dim: int = 10,
        bin_search_tol: float = 0.1,
        lambda_geoda: float = 0.6,
        sigma_geoda: float = 0.0002,
        lambda_fadv=0.0,
        layers_fadv=[1],
        thr_lowpro: float = 0.5,
        lambda_lowpro: float = 1.5,
        eta_lowpro: float = 0.2,
        eta_lowpro_decay: float = 0.98,
        eta_lowpro_min: float = 1e-7,
        eta_newton: float = 0.01,
        targeted=False,
        num_samples=1,
        eps_scale=1,
        batch_size=1,
    ):

        if attack_type not in ["feature-adv"]:
            eps = eps * eps_scale
            eps_step = eps_step * eps_scale
            min_eps = min_eps * eps_scale
            delta = delta * eps_scale

        attack_l12 = set(["fgm", "pgd", "auto-pgd", "wasserstein"])
        if attack_type in attack_l12:
            if norm == 1:
                eps = eps * num_samples
                eps_step = eps_step * num_samples
                if min_eps is not None:
                    min_eps = min_eps * num_samples

            elif norm == 2 or attack_type in ["wasserstein"]:
                eps = eps * np.sqrt(num_samples)
                eps_step = eps_step * np.sqrt(num_samples)
                if min_eps is not None:
                    min_eps = min_eps * np.sqrt(num_samples)

        if attack_type == "boundary":
            return attacks.BoundaryAttack(
                model,
                targeted=targeted,
                delta=delta,
                epsilon=eps,
                step_adapt=step_adapt,
                max_iter=max_iter,
                num_trial=num_trial,
                sample_size=sample_size,
                init_size=init_size,
                min_epsilon=min_eps,
            )

        if attack_type == "hop-skip-jump":
            return attacks.HopSkipJump(
                model,
                targeted=targeted,
                norm=norm,
                max_iter=max_iter,
                max_eval=max_eval,
                init_eval=init_eval,
                init_size=init_size,
            )

        if attack_type == "brendel":
            return attacks.BrendelBethgeAttack(
                model,
                norm=norm,
                targeted=targeted,
                overshoot=overshoot,
                steps=max_iter,
                lr=lr,
                lr_decay=lr_decay,
                lr_num_decay=lr_num_decay,
                momentum=momentum,
                binary_search_steps=binary_search_steps,
                init_size=init_size,
                batch_size=batch_size,
            )

        if attack_type == "deepfool":
            return attacks.DeepFool(
                model,
                max_iter=max_iter,
                epsilon=eps,
                nb_grads=num_grads,
                batch_size=batch_size,
            )

        if attack_type == "elasticnet":
            return attacks.ElasticNet(
                model,
                confidence=confidence,
                targeted=targeted,
                learning_rate=lr,
                binary_search_steps=binary_search_steps,
                max_iter=max_iter,
                beta=beta,
                initial_const=initial_c,
                batch_size=batch_size,
                decision_rule=decision_rule,
            )

        if attack_type == "feature-adv":
            return attacks.FeatureAdversariesPyTorch(
                model,
                delta=delta,
                lambda_=lambda_fadv,
                layer=tuple(layers_fadv),
                max_iter=max_iter,
                batch_size=batch_size,
                step_size=eps_step,
                random_start=num_random_init > 0,
            )

        if attack_type == "threshold":
            return attacks.ThresholdAttack(model, th=th, es=es, targeted=targeted)

        if attack_type == "fgm":
            return attacks.FastGradientMethod(
                model,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                targeted=targeted,
                num_random_init=num_random_init,
                minimal=minimal,
                batch_size=batch_size,
            )

        if attack_type == "bim":
            return attacks.BasicIterativeMethod(
                model,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                batch_size=batch_size,
            )

        if attack_type == "pgd":
            return attacks.ProjectedGradientDescentPyTorch(
                model,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                random_eps=random_eps,
                batch_size=batch_size,
            )

        if attack_type == "auto-pgd":
            if len(model.input_shape) == 1:
                # autopgd only works with image kind shape
                model._input_shape = (1, 1, model.input_shape[0])
            attack = attacks.AutoProjectedGradientDescent(
                model,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                nb_random_init=max(1, num_random_init),
                batch_size=batch_size,
            )
            attack.generate = make_4d_hook(attack.generate)
            return attack

        if attack_type == "auto-cgd":
            if len(model.input_shape) == 1:
                # autopgd only works with image kind shape
                model._input_shape = (1, 1, model.input_shape[0])
            attack = attacks.AutoConjugateGradient(
                model,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                nb_random_init=max(1, num_random_init),
                batch_size=batch_size,
            )
            attack.generate = make_4d_hook(attack.generate)
            return attack

        if attack_type == "geoda":
            return attacks.GeoDA(
                model,
                norm=norm,
                sub_dim=sub_dim,
                max_iter=max_iter,
                bin_search_tol=bin_search_tol,
                lambda_param=lambda_geoda,
                sigma=sigma_geoda,
                batch_size=batch_size,
            )

        if attack_type == "jsma":
            return attacks.SaliencyMapMethod(
                model, theta=theta, gamma=gamma, batch_size=batch_size
            )

        if attack_type == "low-pro-fool":
            return attacks.LowProFool(
                model,
                n_steps=max_iter,
                threshold=thr_lowpro,
                lambd=lambda_lowpro,
                eta=eta_lowpro,
                eta_decay=eta_lowpro_decay,
                eta_min=eta_lowpro_min,
                norm=norm,
            )

        if attack_type == "newtonfool":
            return attacks.NewtonFool(
                model, eta=eta_newton, max_iter=max_iter, batch_size=batch_size
            )

        if attack_type == "cw-l2":
            return attacks.CarliniL2Method(
                model,
                confidence,
                learning_rate=lr,
                binary_search_steps=binary_search_steps,
                max_iter=max_iter,
                targeted=targeted,
                initial_const=initial_c,
                max_halving=max_halving,
                max_doubling=max_doubling,
                batch_size=batch_size,
            )

        if attack_type == "cw-linf":
            return attacks.CarliniLInfMethod(
                model,
                confidence,
                learning_rate=lr,
                max_iter=max_iter,
                targeted=targeted,
                decrease_factor=tau_decr_factor,
                initial_const=initial_c,
                largest_const=largest_c,
                const_factor=c_incr_factor,
                batch_size=batch_size,
            )

        if attack_type == "zoo":
            return attacks.ZooAttack(
                model,
                confidence,
                learning_rate=lr,
                max_iter=max_iter,
                initial_const=initial_c,
                targeted=targeted,
                binary_search_steps=binary_search_steps,
                abort_early=abort_early,
                use_resize=False,
                use_importance=use_importance,
                nb_parallel=num_parallel,
                variable_h=variable_h,
                batch_size=batch_size,
            )

        if attack_type == "shadow":
            if len(model.input_shape) == 1:
                # autopgd only works with image kind shape
                model._input_shape = (1, 1, model.input_shape[0])

            attack = attacks.ShadowAttack(
                model,
                sigma=sigma,
                nb_steps=max_iter,
                learning_rate=lr,
                lambda_tv=lambda_tv,
                lambda_c=lambda_c,
                lambda_s=lambda_s,
                batch_size=batch_size,
                targeted=targeted,
            )
            attack.generate = make_4d_hook(attack.generate)
            return attack

        if attack_type == "wasserstein":
            if len(model.input_shape) == 1:
                # autopgd only works with image kind shape
                model._input_shape = (1, 1, model.input_shape[0])

            attack = attacks.Wasserstein(
                model,
                targeted=targeted,
                p=p_wassertein,
                regularization=reg,
                kernel_size=kernel_size,
                eps=eps,
                eps_step=eps_step,
                eps_factor=eps_factor,
                eps_iter=eps_iter,
                max_iter=max_iter,
                conjugate_sinkhorn_max_iter=conj_sinkhorn_iter,
                projected_sinkhorn_max_iter=proj_sinkhorn_iter,
                batch_size=batch_size,
            )
            attack.generate = make_4d_hook(attack.generate)
            return attack

        raise Exception("%s is not a valid attack type" % (attack_type))

    @staticmethod
    def filter_args(**kwargs):

        if "no_abort" in kwargs:
            kwargs["abort_early"] = not kwargs["no_abort"]

        if "norm" in kwargs:
            if kwargs["norm"] == "inf":
                kwargs["norm"] = np.inf
            else:
                kwargs["norm"] = int(kwargs["norm"])

        args = filter_func_args(ARTAttackFactory.create, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--attack-type",
            type=str.lower,
            default="fgm",
            choices=[
                "boundary",
                "brendel",
                "deepfool",
                "fgm",
                "bim",
                "pgd",
                "auto-pgd",
                "auto-cgd",
                "feature-adv",
                "low-pro-fool",
                "jsma",
                "newtonfool",
                "cw-l2",
                "cw-linf",
                "elasticnet",
                "hop-skip-jump",
                "zoo",
                "threshold",
                "shadow",
                "wasserstein",
            ],
            help=("Attack type"),
        )

        parser.add_argument(
            "--norm",
            type=str.lower,
            default="inf",
            choices=["inf", "1", "2"],
            help=("Attack norm"),
        )

        parser.add_argument(
            "--eps",
            default=0,
            type=float,
            help=("attack epsilon, upper bound for the perturbation norm"),
        )

        parser.add_argument(
            "--eps-step",
            default=0.1,
            type=float,
            help=("Step size of input variation for minimal perturbation computation"),
        )

        parser.add_argument(
            "--delta",
            default=0.1,
            type=float,
            help=("Initial step size for the orthogonal step in boundary-attack"),
        )

        parser.add_argument(
            "--step-adapt",
            default=0.667,
            type=float,
            help=(
                "Factor by which the step sizes are multiplied or divided, "
                "must be in the range (0, 1)."
            ),
        )

        parser.add_argument(
            "--confidence",
            default=0,
            type=float,
            help=("confidence for carlini-wagner attack"),
        )

        parser.add_argument(
            "--lr",
            default=1e-2,
            type=float,
            help=("learning rate for attack optimizers"),
        )

        parser.add_argument(
            "--lr-decay",
            default=0.5,
            type=float,
            help=("learning rate decay for attack optimizers"),
        )

        parser.add_argument(
            "--lr-num-decay",
            default=10,
            type=int,
            help=("learning rate decay steps for attack optimizers"),
        )

        parser.add_argument(
            "--momentum",
            default=0.8,
            type=float,
            help=("momentum for attack optimizers"),
        )

        parser.add_argument(
            "--overshoot",
            default=1.1,
            type=float,
            help=("overshoot param. for Brendel attack"),
        )

        parser.add_argument(
            "--binary-search-steps",
            default=9,
            type=int,
            help=("num bin. search steps in carlini-wagner-l2 attack"),
        )

        parser.add_argument(
            "--max-iter",
            default=10,
            type=int,
            help=("max. num. of optim iters in attack"),
        )

        parser.add_argument(
            "--num-trial",
            default=25,
            type=int,
            help=("Maximum number of trials per iteration (boundary attack)."),
        )

        parser.add_argument(
            "--num-grads",
            default=10,
            type=int,
            help=("number of class gradients (deepfool attack)."),
        )

        parser.add_argument(
            "--sample-size",
            default=20,
            type=int,
            help=("Number of samples per trial (boundary attack)."),
        )

        parser.add_argument(
            "--init-size",
            default=100,
            type=int,
            help=(
                "Maximum number of trials for initial generation of "
                "adversarial examples. (boundary attack)."
            ),
        )

        parser.add_argument(
            "--init-eval",
            default=100,
            type=int,
            help=("Initial number of evaluations for estimating gradient."),
        )

        parser.add_argument(
            "--max-eval",
            default=10000,
            type=int,
            help=("Maximum number of evaluations for estimating gradient."),
        )

        parser.add_argument(
            "--num-random-init",
            default=0,
            type=int,
            help=(
                "Number of random initialisations within the epsilon ball. "
                "For random_init=0 starting at the original input."
            ),
        )

        parser.add_argument(
            "--minimal",
            default=False,
            action="store_true",
            help=(
                "Indicates if computing the minimal perturbation (True). "
                "If True, also define eps_step for the step size and eps "
                "for the maximum perturbation."
            ),
        )

        parser.add_argument(
            "--random-eps",
            default=False,
            action="store_true",
            help=(
                "When True, epsilon is drawn randomly from "
                "truncated normal distribution. "
                "The literature suggests this for FGSM based training to "
                "generalize across different epsilons. eps_step is modified "
                "to preserve the ratio of eps / eps_step. "
                "The effectiveness of this method with PGD is untested"
            ),
        )

        parser.add_argument(
            "--min-eps",
            default=1e-6,
            type=float,
            help=("Stop attack if perturbation is smaller than min_eps."),
        )

        parser.add_argument(
            "--theta",
            default=0.1,
            type=float,
            help=(
                "Amount of Perturbation introduced to each modified "
                "feature per step (can be positive or negative)."
            ),
        )

        parser.add_argument(
            "--gamma",
            default=1.0,
            type=float,
            help=("Maximum fraction of features being perturbed (between 0 and 1)."),
        )

        parser.add_argument(
            "--beta",
            default=0.001,
            type=float,
            help=("Hyperparameter trading off L2 minimization for L1 minimization"),
        )

        parser.add_argument(
            "--decision-rule",
            default="EN",
            choices=["EN", "L1", "L2"],
            help=(
                "Decision rule. ‘EN’ means Elastic Net rule, ‘L1’ means L1 rule, ‘L2’ means L2 rule. (elasticnet)"
            ),
        )

        parser.add_argument(
            "--eta", default=0.01, type=float, help=("Eta coeff. for NewtonFool")
        )

        parser.add_argument(
            "--initial-c",
            default=1e-2,
            type=float,
            help=("Initial weight of constraint function f in carlini-wagner attack"),
        )

        parser.add_argument(
            "--largest-c",
            default=20.0,
            type=float,
            help=("largest weight of constraint function f in carlini-wagner attack"),
        )

        parser.add_argument(
            "--c-incr-factor",
            default=2,
            type=float,
            help=("factor to increment c in carline-wagner-l0/inf attack"),
        )

        parser.add_argument(
            "--tau-decr-factor",
            default=0.9,
            type=float,
            help=("factor to reduce tau in carline-wagner-linf attack"),
        )

        parser.add_argument(
            "--max-halving",
            default=5,
            type=int,
            help=("Maximum number of halving steps in the line search optimization."),
        )

        parser.add_argument(
            "--max-doubling",
            default=5,
            type=int,
            help=("Maximum number of doubling steps in the line search optimization."),
        )

        parser.add_argument(
            "--abort-early",
            default=True,
            action=ActionYesNo,
            help=("abort early in optimizer iterations"),
        )

        parser.add_argument(
            "--use-importance",
            default=False,
            action="store_true",
            help=("to use importance sampling when choosing coordinates to update."),
        )

        parser.add_argument(
            "--variable-h",
            default=0.0001,
            type=float,
            help=("Step size for numerical estimation of derivatives."),
        )

        parser.add_argument(
            "--num-parallel",
            default=128,
            type=int,
            help=("Number of coordinate updates to run in parallel"),
        )

        parser.add_argument(
            "--th",
            default=None,
            type=int,
            help=(
                "Threshold for threshold attack, None indicates finding and minimum threshold"
            ),
        )
        parser.add_argument(
            "--es",
            default=0,
            type=int,
            help=(
                "Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy"
            ),
        )

        parser.add_argument(
            "--sigma",
            default=0.5,
            type=float,
            help=("Standard deviation random Gaussian Noise"),
        )

        parser.add_argument(
            "--lambda-tv",
            default=0.3,
            type=float,
            help=(
                "Scalar penalty weight for total variation of the perturbation (shadow)"
            ),
        )

        parser.add_argument(
            "--lambda-c",
            default=1.0,
            type=float,
            help=(
                "Scalar penalty weight for change in the mean of each color channel of the perturbation"
            ),
        )

        parser.add_argument(
            "--lambda-s",
            default=0.5,
            type=float,
            help=(
                "Scalar penalty weight for similarity of color channels in perturbation"
            ),
        )
        parser.add_argument(
            "--lambda-fadv",
            default=0.0,
            type=float,
            help=("Regularization parameter of the L-inf soft constraint"),
        )
        parser.add_argument(
            "--layers-fadv",
            default=[1],
            type=int,
            nargs="+",
            help=("indices of the representation layers"),
        )

        parser.add_argument(
            "--reg",
            default=3000,
            type=float,
            help=("Entropy regularization.(wasserstein)"),
        )

        parser.add_argument(
            "--kernel-size",
            default=5,
            type=int,
            help=("Kernel size for computing the cost matrix"),
        )
        parser.add_argument(
            "--eps-factor",
            default=1.1,
            type=float,
            help=("Factor to increase the epsilon"),
        )
        parser.add_argument(
            "--eps-iter",
            default=10,
            type=int,
            help=("Number of iterations to increase the epsilon."),
        )
        parser.add_argument(
            "--p-wassertein",
            default=2,
            type=int,
            help=("lp distance for wassertein distance"),
        )
        parser.add_argument(
            "--conj-sinkhorn-iter",
            default=400,
            type=int,
            help=("maximum number of iterations for the conjugate sinkhorn optimizer"),
        )
        parser.add_argument(
            "--proj-sinkhorn-iter",
            default=400,
            type=int,
            help=("maximum number of iterations for the projected sinkhorn optimizer"),
        )

        parser.add_argument(
            "--thr-lowpro",
            type=float,
            default=0.5,
            help="""Lowest prediction probability of a valid adversary for low-pro-fool""",
        )
        parser.add_argument(
            "--lambda-lowpro",
            type=float,
            default=1.5,
            help="""Amount of lp-norm impact on objective function for low-pro-fool""",
        )
        parser.add_argument(
            "--eta-lowpro",
            type=float,
            default=0.2,
            help="""Rate of updating the perturbation vectors for low-pro-fool""",
        )
        parser.add_argument(
            "--eta-lowpro-decay",
            type=float,
            default=0.98,
            help="""Step-by-step decrease of eta for low-pro-fool""",
        )
        parser.add_argument(
            "--eta-lowpro-min", type=float, default=1e-7, help="""Minimal eta value"""
        )
        parser.add_argument(
            "--eta-newton", type=float, default=0.01, help="""eta for newtonfool"""
        )
        # parser.add_argument(
        #     "--sub-dim",
        #     default=10,
        #     type=int,
        #     help="Dimensionality of 2D frequency space (DCT).",
        # )

        # parser.add_argument(
        #     "--bin-search-tol",
        #     default=0.1,
        #     type=float,
        #     help="""Maximum remaining L2 perturbation defining binary search
        #     convergence""",
        # )
        # parser.add_argument(
        #     "--lambda-geoda",
        #     default=0.6,
        #     type=float,
        #     help="""The lambda of equation 19 with lambda_param=0 corresponding to a
        #     single iteration and lambda_param=1 to a uniform distribution of
        #     iterations per step.""",
        # )
        # parser.add_argument(
        #     "--sigma-geoda",
        #     default=0.0002,
        #     type=float,
        #     help="""Variance of the Gaussian perturbation.""",
        # )

        parser.add_argument(
            "--targeted",
            default=False,
            action="store_true",
            help="use targeted attack intead of non-targeted",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='ART attack options')

    add_argparse_args = add_class_args
