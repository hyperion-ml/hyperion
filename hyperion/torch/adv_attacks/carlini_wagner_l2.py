"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .adv_attack import AdvAttack

class CarliniWagnerL2(AdvAttack):

    def __init__(self, model, confidence=0.0, lr=1e-2, 
                 binary_search_steps=9, max_iter=10000,
                 abort_early=True, initial_c=1e-3, 
                 targeted=False, range_min=None, range_max=None):

        super(CarliniWagnerL2, self).__init__(model, None, targeted, range_min, range_max)
        self.confidence = confidence
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.repeat = binary_search_steps >= 10
        self.max_iter = max_iter
        self.abort_early = abort_early
        self.initial_c = initial_c
        self.is_binary = None
        self.box_scale = (self.range_max - self.range_min)/2
        self.box_bias = (self.range_max + self.range_min)/2
        



    @staticmethod
    def atanh(x, eps=1e-6):
        x = (1-eps) * x
        return 0.5 * torch.log((1+x)/(1-x))
    

    def x_w(self, w):
        return self.box_scale * torch.tanh(w) + self.box_bias

    def w_x(self, x):
        return self.atanh((x-self.box_bias)/self.box_scale)


    def f(self, z, target):
        if self.is_binary:
            z_t = z.clone()
            z_t[target==0] *= -1
            z_other = 0
        else:
            z_other = torch.max(z)
            idx = torch.arange(0, z.shape[0], device=z.device)
            z_t = z[idx, target]
            z_clone = z.clone()
            z_clone[idx, target] = -1e10
            z_other = torch.max(z, dim=-1)[0]

        if self.targeted:
            f = F.relu(z_other-z_t+self.confidence) #max(0, z_other-z_target+k)
        else:
            f = F.relu(z_t-z_other+self.confidence) #max(0, z_target-z_other+k)
        return f
        

    def generate(self, input, target):
        
        if self.is_binary is None:
            # run the model to know weather is binary classification problem or multiclass
            z = self.model(input)
            if z.shape[-1] == 1:
                self.is_binary = True
            else:
                self.is_binary = None
            del z

        norm_dim = tuple([i for i in range(1,input.dim())])

        w0 = self.w_x(input).detach() #transform x into tanh space
        
        batch_size = input.shape[0]
        global_best_norm = 1e10*torch.ones(batch_size, device=input.device)
        global_success = torch.zeros(batch_size, dtype=torch.uint8, device=input.device)
        best_adv = input.clone()

        c_lower_bound = torch.zeros(batch_size, device=w0.device)
        c_upper_bound = 1e10 * torch.ones(batch_size, device=w0.device)
        c = self.initial_c * torch.ones(batch_size, device=w0.device)

        for bs_step in range(self.binary_search_steps):

            if self.repeat and bs_step == self.binary_search_steps-1:
                # The last iteration (if we run many steps) repeat the search once.
                c = c_upper_bound

            delta = 1e-3 * torch.randn_like(w0).detach()
            delta.requires_grad = True
            opt = optim.Adam([delta], lr=self.lr)
            loss_prev = 1e10
            best_norm = 1e10*torch.ones(batch_size, device=w0.device)
            success = torch.zeros(batch_size, dtype=torch.uint8, device=w0.device)
            for opt_step in range(self.max_iter):

                opt.zero_grad()

                d_norm = torch.norm(delta, dim=norm_dim)
                w = w0 + delta
                x_adv = self.x_w(w)
                z = self.model(x_adv)
                f = self.f(z, target)
                loss1 = d_norm.mean()
                loss2 = (c * f).mean()
                loss = loss1 + loss2

                loss.backward()
                opt.step()

                #if the attack is successful f(x+delta)==0
                step_success = (f == 0)

                if opt_step % (self.max_iter//10) == 0:
                    logging.info('carlini-wagner bin-search-step={0:d}, opt-step={1:d} '
                                 'loss={2:.2f} d_norm={3:.2f} cf={4:.2f}'.format(
                                     bs_step, opt_step,
                                     loss.item(), loss1.item(), loss2.item()))
                
                    loss_it = loss.item()
                    if self.abort_early:
                        if loss_it > 0.999*loss_prev:
                            break
                        loss_prev = loss_it

                #find elements that reduced l2 and where successful for current c value
                improv_idx = (d_norm < best_norm) & step_success
                best_norm[improv_idx] = d_norm[improv_idx]
                success[improv_idx] = 1

                #find elements that reduced l2 and where successful for global optimization
                improv_idx = (d_norm < global_best_norm) & step_success
                global_best_norm[improv_idx] = d_norm[improv_idx]
                global_success[improv_idx] = 1
                best_adv[improv_idx] = x_adv[improv_idx]
                
            #readjust c
            c_upper_bound[success] = torch.min(c_upper_bound[success], c[success])
            c_lower_bound[~success] = torch.max(c_lower_bound[~success], c[~success])
            avg_c_idx = c_upper_bound < 1e9
            c[avg_c_idx] = (c_lower_bound[avg_c_idx] + c_upper_bound[avg_c_idx])/2
            cx10_idx = (~success) & (~avg_c_idx)
            c[cx10_idx] *= 10
            
        return best_adv
