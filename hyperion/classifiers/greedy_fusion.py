"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import numpy as np

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel
from ..metrics import dcf

from .binary_logistic_regression import BynaryLogisticRegression as BLR

class GreedyFusionBinaryLR(HypModel):

    def __init__(self, weights=None, bias=None,
                 system_idx=None, system_names=None, max_systems=None,
                 prioritize_positive=True,
                 penalty='l2', lambda_reg=1e-6,
                 bias_scaling=1, prior=0.5, prior_eval=None,
                 solver='liblinear', max_iter=100,
                 dual=False, tol=0.0001, verbose=0, lr_seed=1024, **kwargs):

        super(GreedyFusionBinaryLR, self).__init__(**kwargs)

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
        
        self.lr = BLR(penalty=penalty, lambda_reg=lambda_reg,
                      use_bias=True, bias_scaling=bias_scaling,
                      prior=prior, solver=solver, max_iter=max_iter,
                      dual=dual, tol=tol, verbose=verbose, warm_start=False,
                      lr_seed=lr_seed)
        


    def get_fusion_params(self, idx):
        return self.weights[idx], self.bias[idx], self.system_idx[idx]

    
    def fit(self, x, class_ids, sample_weights=None):
        
        num_systems = x.shape[1]
        if self.max_systems = None:
            self.max_systems = min(10, num_systems)

        self.weights = []
        self.bias = []
        self.system_idx = []
        fus_min_dcf = np.zeros((self.max_systems,), dtype=float_cpu())
        fus_act_dcf = np.zeros((self.max_systems,), dtype=float_cpu())
        for i in range(self.max_systems):
            cand_systems = np.arange(num_systems)
            fixed_systems = []
            if i > 0:
                fixed_systems = system_idx[i-1]
                cand_systems[fixed_systems] = -1
                cand_systems = cand_systems[cand_systems>-1]
            
            num_cands = len(cand_systems)
            cand_min_dcf = np.zeros((num_cands,), dtype=float_cpu())
            cand_act_dcf = np.zeros((num_cands,), dtype=float_cpu())
            all_pos = np.zeros((num_cands,), dtype=np.bool)
            cand_weights = []
            for j in range(num_cands):
                system_idx = np.concatenate((fixed_systems, cand_systems), axis=0)
                x_ij = x[:, system_idx]
                self.lr.fit(x_ij, class_ids)
                cand_weights.append([self.lr.A, self.lr.b])
                all_pos[j] = np.all(self.lr.A > 0)
                
                y_ij = self.lr.predict(x_ij)
                tar = y_ij[class_ids==1]
                non = y_ij[class_ids==0]
                min_dcf, act_dcf, _, _ = dcf.fast_eval_dcf_eer(
                    tar, non, self.prior_eval)
                cand_min_dcf[j] = np.mean(min_dcf)
                cand_act_dcf[j] = np.mean(act_dcf)
                
                fus_name = self._make_fus_name(system_idx)
                logging.info('fus_sys=%s min_dcf=%.3f act_dcf=%.3f' % (
                        fus_name, cand_min_dcf[j], cand_act_dcf[j]))
            
            dcf_best = 100
            if self.prioritize_positive:
                allpos_cand_act_dcf = cand_act_dcf
                allpos_cand_act_dcf[all_pos==False] = 100
                j_best = np.argmin(allpos_cand_act_dcf)
                dcf_best = allpos_cand_act_dcf[j_best]
            
            if dcf_best == 100:
                j_best = np.argmin(cand_act_dcf)
                dcf_best = cand_act_dcf[j_best]
                
            select_system = cand_systems[j_best]
            if i==0:
                fus_system_i = np.array([select_system])
            else:
                fus_system_i = np.concatenate((self.system_idx[i-1], [select_system]))

            self.system_idx.append(fus_system_i)

            weights_i, bias_i = cand_weights[j_best]
            self.weights.append(weights_i)
            self.bias.append(bias_i)
            fus_min_dcf[i] = cand_min_dcf[j_best]
            fus_act_dcf[i] = cand_act_dcf[j_best]

        # print report
        for i in range(self.max_systems):
            fus_name = self._make_fus_name(self.systems_idx)
            weigths_str = np.array2string(self.weights[i], separator=',')
            bias_str = np.array2string(self.bias[i], separator=',')
            logging.info('Best-%d=%s min_dcf=%.3f act_dcf=%.3f w' % (
                    i,fus_name,fus_min_dcf[i],fus_act_dcf[i], weigths_str, bias_str))
        
        return fus_min_dcf, fus_act_dcf

                
    def _make_fus_name(self, idx):
        sys_names = [self.system_names[i] for i idx]
        fus_name = '+'.join(sys_names)
        return fus_name
                
