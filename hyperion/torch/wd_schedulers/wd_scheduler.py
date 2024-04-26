"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.optim as optim


class WDScheduler:
    """Base class for weight decay schedulers.

    Attributes:
      optimizer: Pytorch optimizer object.
      initial_wd: initial value of the weight decay.
      warmup_steps: number of warm up steps to get the the weight decay to its final value.
      epoch: initial training training epoch, this is needed to restart the model
             training.
      step: initial training step, this is needed to restart the model training.
      update_wd_on_opt_step: if True, updates the weight decay each time we update the model,
        otherwise after each epoch.
    """

    def __init__(
        self,
        optimizer,
        initial_wd=0,
        warmup_steps=0,
        epoch=0,
        step=0,
        update_wd_on_opt_step=False,
    ):
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("%s is not an Optimizer" % (type(optimizer).__name__))
        self.optimizer = optimizer

        if epoch == 0:
            for group in optimizer.param_groups:
                group.setdefault("final_wd", group["weight_decay"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "final_wd" not in group:
                    raise KeyError(
                        "param 'final_wd' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )

        self.final_wds = list(
            map(lambda group: group["final_wd"], optimizer.param_groups)
        )

        if isinstance(initial_wd, list) or isinstance(initial_wd, tuple):
            if len(initial_wd) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} initial_wds, got {}".format(
                        len(optimizer.param_groups), len(initial_wd)
                    )
                )
            self.initial_wds = list(initial_wd)
        else:
            max_wd = max([group["final_wd"] for group in optimizer.param_groups])
            self.initial_wds = [
                initial_wd * group["final_wd"] / max_wd
                for group in optimizer.param_groups
            ]

        if epoch == 0:
            for group, wd in zip(optimizer.param_groups, self.initial_wds):
                group["weight_decay"] = wd

        self.warmup_steps = warmup_steps
        self.epoch = epoch
        self.step = step
        self.update_wd_on_opt_step = update_wd_on_opt_step

    @property
    def in_warmup(self):
        return self.step < self.warmup_steps

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_wd(self):
        raise NotImplementedError

    def on_epoch_begin(self, epoch=None, **kwargs):
        if epoch is not None:
            self.epoch = epoch

        if self.update_wd_on_opt_step:
            return

        for param_group, wd in zip(
            self.optimizer.param_groups, self.get_wd(self.epoch)
        ):
            param_group["weight_decay"] = wd

    def on_epoch_end(self, metrics=None):
        self.epoch += 1

    def on_opt_step(self):
        if self.update_wd_on_opt_step:
            for param_group, wd in zip(
                self.optimizer.param_groups, self.get_wd(self.step)
            ):
                param_group["weight_decay"] = wd

        self.step += 1
