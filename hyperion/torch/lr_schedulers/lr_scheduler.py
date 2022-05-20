"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import torch
import torch.optim as optim


class LRScheduler(object):
    """Base class for learning rate schedulers.

    Attributes:
      optimizer: Pytorch optimizer object.
      min_lr: minimum learning rate.
      warmup_steps: number of warm up steps to get the lr from 0 to the maximum lr.
      epoch: initial training training epoch, this is needed to restart the model
             training.
      step: initial training step, this is needed to restart the model training.
      update_lr_on_opt_step: if True, updates the lr each time we update the model,
        otherwise after each epoch.
    """

    def __init__(
        self,
        optimizer,
        min_lr=0,
        warmup_steps=0,
        epoch=0,
        step=0,
        update_lr_on_opt_step=False,
    ):
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("%s is not an Optimizer" % (type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if epoch == 0:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )

        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.warmup_steps = warmup_steps
        self.epoch = epoch
        self.step = step
        self.update_lr_on_opt_step = update_lr_on_opt_step

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

    def get_warmup_lr(self):
        x = self.step
        return [
            (base_lr - min_lr) / self.warmup_steps * x + min_lr
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

    def get_lr(self):
        raise NotImplementedError

    def on_epoch_begin(self, epoch=None, **kwargs):
        if epoch is not None:
            self.epoch = epoch

        if self.update_lr_on_opt_step:
            return

        for param_group, lr in zip(
            self.optimizer.param_groups, self.get_lr(self.epoch)
        ):
            param_group["lr"] = lr

    def on_epoch_end(self, metrics=None):
        self.epoch += 1

    def on_opt_step(self):

        if self.in_warmup:
            for param_group, lr in zip(
                self.optimizer.param_groups, self.get_warmup_lr()
            ):
                param_group["lr"] = lr
            self.step += 1
            return

        if self.update_lr_on_opt_step:
            for param_group, lr in zip(
                self.optimizer.param_groups, self.get_lr(self.step)
            ):
                param_group["lr"] = lr

        self.step += 1
