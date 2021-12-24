"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np

import logging
from abc import ABCMeta, abstractmethod

from ...hyp_defs import float_cpu
from ...utils.math import softmax, logsumexp
from ...utils.queues import GeneratorQueue
from ..core import PDF


class ExpFamilyMixture(PDF):
    __metaclass__ = ABCMeta

    def __init__(
        self, num_comp=1, pi=None, eta=None, min_N=0, update_pi=True, **kwargs
    ):
        super().__init__(**kwargs)
        if pi is not None:
            num_comp = len(pi)
        self.num_comp = num_comp
        self.pi = pi
        self.eta = eta
        self.min_N = min_N
        self.A = None
        self._log_pi = None
        self.update_pi = update_pi

    @property
    def is_init(self):
        if not self._is_init:
            if self.eta is not None and self.A is not None and self.pi is not None:
                self.validate()
                self._is_init = True
        return self._is_init

    @property
    def log_pi(self):
        if self._log_pi is None:
            self._log_pi = np.log(self.pi + 1e-15)
        return self._log_pi

    def _validate_pi(self):
        assert len(self.pi) == self.num_comp

    def fit(
        self,
        x,
        sample_weight=None,
        x_val=None,
        sample_weight_val=None,
        epochs=10,
        batch_size=None,
    ):

        if not self.is_init:
            self.initialize(x)

        log_h = self.accum_log_h(x, sample_weight)
        if x_val is not None:
            log_h_val = self.accum_log_h(x_val, sample_weight_val)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):
            N, u_x = self.Estep(x=x, sample_weight=sample_weight, batch_size=batch_size)
            elbo[epoch] = self.elbo(None, N=N, u_x=u_x, log_h=log_h)
            self.Mstep(N, u_x)

            if x_val is not None:
                N, u_x = self.Estep(
                    x=x_val, sample_weight=sample_weight_val, batch_size=batch_size
                )
                elbo_val[epoch] = self.elbo(None, N=N, u_x=u_x, log_h=log_h_val)

        if x_val is None:
            return elbo, elbo / x.shape[0]
        else:
            return elbo, elbo / x.shape[0], elbo_val, elbo_val / x.shape[0]

    def fit_generator(
        self,
        generator,
        train_steps,
        epochs=10,
        val_data=None,
        val_steps=0,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):

        do_validation = bool(validation_data)
        val_gen = (
            hasattr(validation_data, "next")
            or hasattr(validation_data, "__next__")
            or isinstance(validation_data, Sequence)
        )
        if val_gen and not validation_steps:
            raise ValueError(
                "When using a generator for validation data, "
                "you must specify a value for "
                "`validation_steps`."
            )

        if do_validation and not val_gen:
            x, u_x_val, sample_weight_val = self.tuple2data(val_data)
            log_h_val = self.accum_log_h(x, sample_weight_val)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):
            N, u_x, log_h = self.Estep_generator(
                generator,
                train_steps,
                return_log_h=True,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
            )

            self.Mstep(N, u_x)
            elbo[epoch] = self.elbo(None, N=N, u_x=u_x, log_h=log_h)

            if val_data is not None:
                if val_gen:
                    N, u_x, log_h_val = self.Estep_generator(
                        generator,
                        train_steps,
                        return_log_h=True,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                    )
                else:
                    N, u_x = self.Estep(x_val, u_x_val, sample_weight_val)
                elbo_val[epoch] = self.elbo(None, N=N, u_x=u_x, log_h=log_h_val)

        if x_val is None:
            return elbo, elbo / x.shape[0]
        else:
            return elbo, elbo / x.shape[0], elbo_val, elbo_val / x.shape[0]

    def log_h(self, x):
        return 0

    def accum_log_h(self, x, sample_weight=None):
        if sample_weight is None:
            return np.sum(self.log_h(x))
        return np.sum(sample_weight * self.log_h(x))

    def compute_log_pz(self, x, u_x=None, mode="nat"):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        return np.dot(u_x, self.eta.T) - self.A + self.log_pi

    def compute_pz(self, x, u_x=None, mode="nat"):
        if mode == "nat":
            return self.compute_pz_nat(x, u_x)
        else:
            return self.compute_pz_std(x)

    def compute_pz_nat(self, x, u_x=None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        logr = np.dot(u_x, self.eta.T) - self.A + self.log_pi
        return softmax(logr)

    def compute_pz_std(self, x):
        return self.compute_pz_nat(x)

    def compute_suff_stats(self, x):
        return x

    def accum_suff_stats(self, x, u_x=None, sample_weight=None, batch_size=None):
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_1batch(x, u_x, sample_weight)
        else:
            return self._accum_suff_stats_nbatches(x, sample_weight, batch_size)

    def _accum_suff_stats_1batch(self, x, u_x=None, sample_weight=None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        z = self.compute_pz_nat(x, u_x)
        if sample_weight is not None:
            z *= sample_weight[:, None]

        N = np.sum(z, axis=0)
        acc_u_x = np.dot(z.T, u_x)
        # L_z=gmm.ElnP_z_w(N,gmm.lnw)-gmm.Elnq_z(z);
        return N, acc_u_x

    def _accum_suff_stats_nbatches(self, x, sample_weight, batch_size):
        sw_i = None
        for i1 in range(0, x.shape[0], batch_size):
            i2 = np.minimum(i1 + batch_size, x.shape[0])
            x_i = x[i1:i2, :]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_1batch(x_i, sample_weight=sw_i)
            if i1 == 0:
                N = N_i
                u_x = u_x_i
            else:
                N += N_i
                u_x += u_x_i
        return N, u_x

    def accum_suff_stats_segments(
        self, x, segments, u_x=None, sample_weight=None, batch_size=None
    ):
        K = self.num_comp
        num_segments = len(segments)
        N = np.zeros((num_segments, K), dtype=float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), dtype=float_cpu())
        u_x_i = None
        sw_i = None
        for i in range(num_segments):
            start = int(segments[i][0])
            end = int(segments[i][1]) + 1
            x_i = x[start:end]
            if u_x is not None:
                u_x_i = u_x[start:end]
            if sample_weight is not None:
                sw_i = sample_weight[start:end]
            N_i, acc_u_x_i = self.accum_suff_stats(
                x_i, u_x=u_x_i, sample_weight=sw_i, batch_size=batch_size
            )
            N[i] = N_i
            acc_u_x[i] = acc_u_x_i

        return N, acc_u_x

    def accum_suff_stats_segments_prob(
        self, x, prob, u_x=None, sample_weight=None, batch_size=None
    ):
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_segments_prob_1batch(
                x, prob, u_x, sample_weight
            )
        else:
            return self._accum_suff_stats_segments_prob_nbatches(
                x, prob, sample_weight, batch_size
            )

    def _accum_suff_stats_segments_prob_1batch(
        self, x, prob, u_x=None, sample_weight=None
    ):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        z = self.compute_pz_nat(x, u_x)
        if sample_weight is not None:
            z *= sample_weight[:, None]

        K = len(self.pi)
        num_segments = prob.shape[1]
        N = np.zeros((num_segments, K), float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), float_cpu())

        for i in range(num_segments):
            z_i = z * prob[:, i][:, None]
            N[i] = np.sum(z_i, axis=0)
            acc_u_x[i] = np.dot(z_i.T, u_x)

        return N, acc_u_x

    def _accum_suff_stats_segments_prob_nbatches(
        self, x, prob, sample_weight, batch_size
    ):

        sw_i = None
        for i1 in range(0, x.shape[0], batch_size):
            i2 = np.minimum(i1 + batch_size, x.shape[0])
            x_i = x[i1:i2, :]
            prob_i = prob[i1:i2, :]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_segments_prob_1batch(
                x_i, prob_i, sample_weight=sw_i
            )
            if i1 == 0:
                N = N_i
                u_x = u_x_i
            else:
                N += N_i
                u_x += u_x_i
        return N, u_x

    def accum_suff_stats_sorttime(
        self,
        x,
        frame_length,
        frame_shift,
        u_x=None,
        sample_weight=None,
        batch_size=None,
    ):
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_sorttime_1batch(
                x, frame_length, frame_shift, u_x, sample_weight
            )
        else:
            return self._accum_suff_stats_sorttime_nbatches(
                x, frame_length, frame_shift, sample_weight, batch_size
            )

    def _accum_suff_stats_sorttime_1batch(
        self, x, frame_length, frame_shift, u_x=None, sample_weight=None
    ):

        K = len(self.pi)
        num_frames = x.shape[0]
        num_segments = int(np.floor((num_frames - frame_length) / frame_shift + 1))
        if num_segments == 1:
            return self._accum_suff_stats_1batch(self, x, u_x, sample_weight)

        if u_x is None:
            u_x = self.compute_suff_stats(x)
        z = self.compute_pz_nat(x, u_x)
        if sample_weight is not None:
            z *= sample_weight[:, None]

        N = np.zeros((num_segments, K), float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), float_cpu())

        start1 = int(frame_shift - 1)
        end1 = int((num_segments - 1) * frame_shift)
        start2 = int(start1 + frame_length)
        end2 = int(end1 + frame_length)
        cum_N = np.cumsum(z, axis=0)
        N[0] = cum_N[frame_length - 1]
        N[1:] = cum_N[start2:end2:frame_shift] - cum_N[start1:end1:frame_shift]

        for k in range(K):
            cum_u_x_k = np.cumsum(z[:, k][:, None] * u_x, axis=0)
            acc_u_x[0, k] = cum_u_x_k[frame_length - 1]
            acc_u_x[1:, k] = (
                cum_u_x_k[start2:end2:frame_shift] - cum_u_x_k[start1:end1:frame_shift]
            )

        return N, acc_u_x

    def _accum_suff_stats_sorttime_nbatches(
        self, x, frame_length, frame_shift, sample_weight, batch_size
    ):

        K = len(self.pi)
        num_frames = x.shape[0]
        num_segments = int(np.floor((num_frames - frame_length) / frame_shift + 1))
        if num_segments == 1:
            return self._accum_suff_stats_1batch(self, x, u_x, sample_weight)

        num_segments_per_batch = np.floor((num_frames - frame_length) / frame_shift + 1)
        batch_size = int((num_segments_per_batch - 1) * frame_shift + frame_length)
        batch_shift = int(num_segments_per_batch * frame_shift)

        N = np.zeros((num_segments, K), float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), float_cpu())

        sw_i = None
        cur_segment = 0
        for i1 in range(0, x.shape[0], batch_shift):
            i2 = np.minimum(i1 + batch_size, x.shape[0])
            x_i = x[i1:i2, :]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_sorttime_1batch(
                x_i, frame_length, frame_shift, sample_weight=sw_i
            )
            num_segments_i = N_i.shape[0]
            N[cur_segment : cur_segment + num_segments_i] = N_i
            acc_u_x[cur_segment : cur_segment + num_segments_i] = u_x_i
            cur_segment += num_segments_i
        return N, acc_u_x

    def Estep(self, x, u_x=None, sample_weight=None, batch_size=None):
        return self.accum_suff_stats(x, u_x, sample_weight, batch_size)

    def Estep_generator(
        self,
        generator,
        num_steps,
        return_log_h,
        max_queue_size=10,
        workers=1,
        use_multiprocessin=False,
    ):
        wait_time = 0.01  # in secs
        queue = None
        N = None
        acc_u_x = None
        log_h = 0
        try:
            queue = GeneratorQueue(
                generator, use_multiprocessing=use_multiprocessing, wait_time=wait_time
            )
            queue.start(workers=workers, max_queue_size=max_queue_size)
            queue_generator = queue.get()

            cur_step = 0
            for cur_step in range(num_steps):
                data = next(queue_generator)
                x, u_x, sample_weight = self.tuple2data(data)
                N_i, u_x_i = self.Estep(x, u_x, sample_weight)
                if return_log_h:
                    log_h += self.accum_log_h(x)
                if cur_step == 0:
                    N = N_i
                    acc_u_x = u_x_i
                else:
                    N += N_i
                    acc_u_x += u_x_i
        finally:
            if enqueuer is not None:
                enqueuer.stop()

        if return_log_h:
            return N, acc_u_x, log_h
        else:
            return N, acc_u_x

    def sum_suff_stats(self, N, u_x):
        assert len(N) == len(u_x)
        acc_N = N[1]
        acc_u_x = u_x[1]
        for i in range(1, len(N)):
            acc_N += N
            acc_u_x += u[i]
        return acc_N, acc_u_x

    @abstractmethod
    def Mstep(self, stats):
        pass

    def elbo(self, x, u_x=None, N=1, log_h=None, sample_weight=None, batch_size=None):
        if u_x is None:
            N, u_x = self.accum_suff_stats(
                x, sample_weight=sample_weight, batch_size=batch_size
            )
        if log_h is None:
            log_h = self.accum_log_h(x, sample_weight=sample_weight)
        return log_h + np.sum(u_x * self.eta) + np.inner(N, self.log_pi - self.A)

    def log_prob(self, x, u_x=None, mode="nat"):
        if mode == "nat":
            return self.log_prob_nat(x, u_x)
        else:
            return self.log_prob_std(x)

    def log_prob_nat(self, x, u_x=None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        llk_k = np.dot(u_x, self.eta.T) - self.A + self.log_pi
        llk = logsumexp(llk_k)
        return self.log_h(x) + llk

    @abstractmethod
    def log_prob_std(self, x):
        pass

    def log_prob_nbest(self, x, u_x=None, mode="nat", nbest_mode="master", nbest=1):
        if mode == "nat":
            return self.log_prob_nbest_nat(x, u_x, nbest_mode=nbest_mode, nbest=nbest)
        else:
            return self.log_prob_std(x, nbest_mode=nbest_mode, nbest=nbest)

    def log_prob_nbest_nat(self, x, u_x=None, nbest_mode="master", nbest=1):

        if u_x is None:
            u_x = self.compute_suff_stats(x)
        if nbest_mode == "master":
            assert isinstance(nbest, int)
            llk_k = np.dot(u_x, self.eta.T) - self.A + self.log_pi
            nbest = np.argsort(llk_k)[: -(nbest + 1) : -1]
            llk_k = llk_k[nbest]
        else:
            llk_k = np.dot(u_x, self.eta[nbest, :].T) - self.A + self.log_pi
        llk = logsumexp(llk_k)
        return self.log_h(x) + llk

    @abstractmethod
    def log_prob_nbest_std(self, x, nbest_mode="master", nbest=1):
        pass

    def get_config(self):
        config = {"min_n": self.min_N, "update_pi": self.update_pi}
        base_config = super(ExpFamilyMixture, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def tuple2data(data):
        if isinstance(data, tuple):
            if len(data) == 2:
                x, u_x = data
                if u_x.ndim == 2:
                    sample_weight = None
                elif u_x.ndim == 1:
                    sample_weight = u_x
                    u_x = None
                else:
                    raise ValueError("Generator output: " + str(data))
            elif len(data) == 3:
                x, u_x, sample_weight = data
            else:
                raise ValueError("Generator output: " + str(data))
        else:
            x = data
            u_x = None
            sample_weight = None
        return x, u_x, sample_weight

    @staticmethod
    def compute_A_nat(eta):
        raise NotImplementedError()

    @staticmethod
    def compute_A_std(params):
        raise NotImplementedError()

    @staticmethod
    def compute_eta(param):
        raise NotImplementedError()

    @staticmethod
    def compute_std(eta):
        raise NotImplementedError()

    @abstractmethod
    def _compute_nat_params(self):
        pass

    @abstractmethod
    def _compute_std_params(self):
        pass
