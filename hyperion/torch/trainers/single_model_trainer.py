"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..loggers import LoggerList
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..optim import OptimizerFactory as OF
from ..torch_model import TorchModel
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF
from .torch_trainer_base import AMPDType, DDPType, TorchTrainerBase


class SingleModelTrainer(TorchTrainerBase):
    """Trainer for a single neural network model with optional SWA and logging.

    Attributes (includes inherited members):
        model (TorchModel): Model instance to optimize.
        optim (torch.optim.Optimizer | Dict[str, Any]): Optimizer or optimizer config.
        lrsched (Optional[LRS]): Learning-rate scheduler object or configuration dict.
        wdsched (Optional[WDS]): Weight-decay scheduler object or configuration dict.
        train_mode (str): Named train-mode exposed by the model (e.g., ``\"full\"``).
        loss (Optional[nn.Module]): Criterion applied to model outputs and targets.
        input_key (str): Key used to fetch model inputs from the dataloader batch.
        target_key (str): Key used to fetch supervision targets from the batch.
        exp_path (Path): Directory where checkpoints, logs, and artifacts are stored.
        num_epochs (int): Total number of epochs to execute unless stopped earlier.
        cur_epoch (int): Epoch index at which to resume training.
        max_steps (Optional[int]): Global step budget overriding ``num_epochs`` if set.
        cur_step (int): Current global optimization step.
        grad_acc_steps (int): Number of gradient accumulation steps before each update.
        eff_batch_size (Optional[int]): Effective batch size after accumulation/DDP.
        val_steps (Optional[int]): Number of steps between validation evaluations.
        val_hours (Optional[float]): Max wall-clock hours between validations.
        save_steps (Optional[int]): Number of steps between checkpoint saves.
        save_hours (Optional[float]): Max wall-clock hours between checkpoint saves.
        device (Union[torch.device, int, None]): Device where the model runs.
        loggers (LoggerList): Active loggers receiving training events.
        ddp (bool): Whether to wrap the model in a DDP variant.
        ddp_type (DDPType): Selected DDP backend (standard, OSS, sharded, FSDP).
        cpu_offload (bool): Enables CPU offload for Fully-Sharded DDP.
        use_amp (bool): Enables automatic mixed precision during forward/backward.
        amp_dtype (torch.dtype): Precision to use when AMP is active.
        log_interval (int): Number of optimizer steps between progress logs.
        use_tensorboard (bool): Whether to enable a TensorBoard logger.
        use_wandb (bool): Whether to enable a Weights&Biases logger.
        wandb (Dict[str, Any]): Additional W&B configuration parameters.
        grad_clip (float): Gradient-norm clipping threshold (<=0 disables).
        grad_clip_norm (Union[str, int]): Norm type used when clipping gradients.
        swa_start (int): Step at which to start SWA averaging (0 means disabled).
        swa_lr (float): Learning rate used during the SWA phase.
        swa_anneal_steps (int): Steps used to anneal the SWA learning rate.
        swa_update_steps (int): Steps between SWA parameter averaging operations.
        bn_update_steps (int): Max steps for the BN statistics refresh after SWA.
    """

    def __init__(
        self,
        model: TorchModel,
        optim: torch.optim.Optimizer,
        lrsched: Optional[LRS] = None,
        wdsched: Optional[WDS] = None,
        train_mode: str = "full",
        loss: Optional[nn.Module] = None,
        exp_path: PathLike = "./train",
        num_epochs: int = 100,
        cur_epoch: int = 0,
        max_steps: Optional[int] = None,
        cur_step: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        val_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        device: Union[torch.device, int, None] = None,
        loggers: Optional[LoggerList] = None,
        ddp: bool = False,
        ddp_type: DDPType = DDPType.DDP,
        cpu_offload: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        log_interval: int = 1000,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, str] = {},
        grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        input_key="x",
        target_key: str = "class_id",
    ) -> None:
        """
        Initializes the single-model trainer and prepares the model for training.

        Args:
            model (TorchModel): Model instance to optimize.
            optim (torch.optim.Optimizer): Optimizer already constructed for the model.
            lrsched (Optional[LRS]): Learning-rate scheduler or configuration dict.
            wdsched (Optional[WDS]): Weight-decay scheduler or configuration dict.
            train_mode (str): Name of the train-mode to activate inside the model.
            loss (Optional[nn.Module]): Criterion applied to model outputs.
            exp_path (PathLike): Directory used to save checkpoints and logs.
            num_epochs (int): Maximum number of epochs to run.
            cur_epoch (int): Epoch index to resume from.
            max_steps (Optional[int]): Optional cap on the total number of steps.
            cur_step (int): Global step to resume from.
            grad_acc_steps (int): Gradient accumulation steps per optimizer update.
            eff_batch_size (Optional[int]): Reference effective batch size.
            val_steps (Optional[int]): Number of steps between validations.
            save_steps (Optional[int]): Number of steps between checkpoint saves.
            device (Union[torch.device, int, None]): Device on which to run the model.
            loggers (Optional[LoggerList]): Collection of loggers to receive events.
            ddp (bool): Whether to wrap the model in DistributedDataParallel.
            ddp_type (DDPType): DDP flavor to use if `ddp` is True.
            cpu_offload (bool): Enables CPU offload for Fully-Sharded DDP.
            use_amp (bool): Enables automatic mixed precision.
            amp_dtype (AMPDType): AMP precision (float16 or bfloat16).
            log_interval (int): Steps between progress log entries.
            use_tensorboard (bool): Enables TensorBoard logging.
            use_wandb (bool): Enables Weights & Biases logging.
            wandb (Dict[str, str]): Additional arguments for W&B initialization.
            grad_clip (float): Gradient norm threshold (<=0 disables clipping).
            grad_clip_norm (Union[str, int]): Norm type for gradient clipping.
            swa_start (int): Step at which to start SWA averaging (0 disables).
            swa_lr (int): Learning rate to use during SWA.
            swa_anneal_steps (int): Steps for annealing SWA learning rate.
            input_key (str): Key used to extract model inputs from the dataloader.
            target_key (str): Key used to extract supervision targets.

        Returns:
            None
        """

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.model = model
        self.optim = optim
        self.lrsched = lrsched
        self.wdsched = wdsched
        self.train_mode = train_mode

        self.input_key = input_key
        self.target_key = target_key

        self.loss = loss
        if self.loss is not None:
            self.loss.to(self.device)

        self.set_train_mode()
        self.prepare_models_for_training()

    def prepare_models_for_training(self) -> None:
        """
        Places the model on the desired device, wraps it for DDP/AMP, and
        instantiates the optimizer and schedulers by delegating to
        :meth:`TorchTrainerBase._prepare_model_for_training`.

        Args:
            None.

        Returns:
            None
        """
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
        ) = self._prepare_model_for_training(
            self.model,
            self.optim,
            self.lrsched,
            self.wdsched,
            self.device,
            self.use_amp,
            self.ddp,
            self.ddp_type,
            self.cpu_offload,
            self.do_swa,
            self.swa_lr,
            self.swa_anneal_steps,
        )
        self.grad_scaler = self.get_grad_scaler(self.use_amp, self.ddp, self.ddp_type)

    def set_train_mode(self) -> None:
        """
        Configures the model's internal training mode and logs a parameter
        summary for visibility.

        Args:
            None.

        Returns:
            None
        """
        self.model.set_train_mode(self.train_mode)
        if self.rank == 0:
            logging.info(f"Model train mode: {self.model.train_mode}")
            logging.info(f"Parmeter summary for the model:")
            self.model.parameter_summary(verbose=True)
            logging.info(f"Parameter list for the model:")
            self.model.print_parameter_list()

    def on_epoch_begin(self) -> None:
        """
        Handles bookkeeping before each epoch begins and notifies active
        schedulers so they can update their state.

        Args:
            None.

        Returns:
            None
        """
        super().on_epoch_begin()

        if self.lr_scheduler is not None:
            # this is needed by cosine scheduler
            self.lr_scheduler.on_epoch_begin(
                self.cur_epoch, epoch_updates=self.save_steps
            )

        if self.wd_scheduler is not None:
            self.wd_scheduler.on_epoch_begin(self.cur_epoch)

    def on_epoch_end(self, logs: Dict[str, Any]) -> None:
        """
        Finalizes epoch-level bookkeeping, including scheduler updates when not
        in the SWA phase.

        Args:
            logs (Dict[str, Any]): Aggregated metrics for the epoch.

        Returns:
            None
        """
        super().on_epoch_end(logs)
        if self.do_swa and self.cur_step >= self.swa_start:
            return

        if self.lr_scheduler is not None:
            self.lr_scheduler.on_epoch_end(logs)
        if self.wd_scheduler is not None:
            self.wd_scheduler.on_epoch_end()

    def on_swa_epoch_begin(self) -> None:
        """
        Activates the SWA model before running the SWA-specific epoch.

        Args:
            None.

        Returns:
            None
        """
        super().on_swa_epoch_begin()
        self.model = self.swa_model.module

    def on_swa_epoch_end(self, logs: Dict[str, Any]) -> None:
        """
        Finishes the SWA epoch and forwards logs to the parent trainer.

        Args:
            logs (Dict[str, Any]): Metrics gathered during the SWA epoch.

        Returns:
            None
        """
        super().on_swa_epoch_end(logs)

    def on_train_loop_begin(self) -> None:
        """
        Sets the model to training mode prior to entering the training loop.

        Args:
            None.

        Returns:
            None
        """
        self.model.train()

    def on_val_loop_begin(self) -> None:
        """
        Sets the model to evaluation mode prior to the validation loop.

        Args:
            None.

        Returns:
            None
        """
        self.model.eval()

    def preprocess_data(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Converts the raw dataloader batch into the structure expected by the
        model and loss, re-labeling keys to ``audio`` and ``target``.

        Args:
            batch_data (Dict[str, Any]): Batch emitted by the dataloader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch dict.
        """
        x_lengths_key = f"{self.input_key}_lengths"
        y_lengths_key = f"{self.target_key}_lengths"
        output_batch_data = {
            "id": batch_data["id"],
            "audio": batch_data[self.input_key],
            "target": batch_data[self.target_key],
        }
        if x_lengths_key in batch_data:
            output_batch_data["audio_lengths"] = batch_data[x_lengths_key]
        if y_lengths_key in batch_data:
            output_batch_data["target_lengths"] = batch_data[y_lengths_key]
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Runs the forward pass, computes the loss, and returns all intermediate
        outputs for downstream metric computation.

        Args:
            batch_data (Dict[str, Any]): Preprocessed batch.

        Returns:
            Tuple[torch.Tensor, Any]: Loss tensor and raw model outputs.
        """
        output = self.model(**batch_data)
        loss = self.loss(output, batch_data["target"])
        return loss, output

    def compute_backward(self, loss: torch.Tensor) -> None:
        """
        Performs the backward pass with gradient scaling when AMP is enabled.

        Args:
            loss (torch.Tensor): Scalar loss tensor produced by ``compute_forward``.

        Returns:
            None
        """
        loss = loss.float()
        self.grad_scaler.scale(loss).backward()

    def zero_grad_optimizers(self) -> None:
        """
        Clears gradients on the primary optimizer.

        Args:
            None.

        Returns:
            None
        """
        self.optimizer.zero_grad()

    def get_lrs(self) -> Dict[str, float]:
        """
        Retrieves learning-rate statistics for the wrapped optimizer.

        Args:
            None.

        Returns:
            Dict[str, float]: Map of scheduler/optimizer group names to LRs.
        """
        return self._get_lrs(self.optimizer)

    def get_wds(self) -> Dict[str, float]:
        """
        Retrieves weight-decay statistics for the wrapped optimizer and
        optional scheduler.

        Args:
            None.

        Returns:
            Dict[str, float]: Map of parameter groups to weight decay values.
        """
        return self._get_wds(self.optimizer, self.wd_scheduler)

    def models_have_bn(self) -> bool:
        """
        Indicates whether the managed model contains BatchNorm layers.

        Args:
            None.

        Returns:
            bool: True if any BatchNorm modules are present.
        """
        return self.model.has_batchnorms()

    def update_models(self) -> Dict[str, float]:
        """
        Applies an optimization step with optional scheduler updates, gradient
        clipping, and AMP scaling logic.

        Args:
            None.

        Returns:
            Dict[str, float]: Diagnostics such as the clipped gradient norm.
        """
        if self.lr_scheduler is not None and not self.in_swa:
            self.lr_scheduler.on_opt_step()

        if self.wd_scheduler is not None:
            self.wd_scheduler.on_opt_step()

        grad_norm = self._update_model_by_optim(
            self.model,
            self.optimizer,
            self.grad_clip,
            self.grad_clip_norm,
            self.use_amp,
            self.grad_scaler,
        )
        self.grad_scaler.update()
        logs = {"grad_norm": grad_norm}
        return logs

    def update_swa_model(self) -> None:
        """
        Periodically updates the SWA weight averages and scheduler once the SWA
        phase has started.

        Args:
            None.

        Returns:
            None
        """
        if (
            self.do_swa
            and self.cur_step >= self.swa_start
            and self.cur_step % self.swa_update_steps == 0
        ):
            self.in_swa = True
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

    def save_checkpoint(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Persists the model, optimizer, and scheduler states to disk using the
        base helper utilities.

        Args:
            logs (Optional[Dict[str, Any]]): Metrics to store alongside the checkpoint.

        Returns:
            None
        """
        if self.ddp and (
            self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
        ):
            # Not sure what this does, just copying from the example in
            # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            # optimizer = cast(OSS, optimizer)
            self.optimizer.consolidate_state_dict()

        if self.rank != 0:
            return

        checkpoint = self.model_checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
            logs=logs,
        )

        self.save_model_checkpoint_to_file("model", checkpoint)

    def save_swa_model(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves the SWA-averaged model state to disk when SWA finishing steps
        are complete.

        Args:
            logs (Optional[Dict[str, Any]]): Metrics to store with the checkpoint.

        Returns:
            None
        """
        if self.rank != 0:
            return

        checkpoint = self.model_checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
            logs=logs,
        )
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_model_ep%04d_%010d.pth" % (
            self.exp_path,
            self.cur_epoch,
            self.cur_step,
        )
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, epoch: int, step: int) -> Optional[Dict[str, Any]]:
        """
        Loads a previously saved checkpoint and restores trainer state.

        Args:
            epoch (int): Epoch index encoded in the checkpoint filename.
            step (int): Global step encoded in the checkpoint filename.

        Returns:
            Optional[Dict[str, Any]]: Logs stored in the checkpoint, if any.
        """
        checkpoint = self.load_model_checkpoint_from_file("model", epoch, step)
        logs = self._load_vars_from_checkpoint(checkpoint)
        self._load_model_state_dicts_from_checkpoint(
            checkpoint,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
        )
        return logs

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filters keyword arguments down to those accepted by ``__init__``.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Subset compatible with ``SingleModelTrainer.__init__``.
        """
        args = filter_func_args(SingleModelTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_optim_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Registers optimizer, LR scheduler, and WD scheduler arguments on an
        :class:`ArgumentParser`.

        Args:
            parser (ArgumentParser): Parser receiving the arguments.
            prefix (Optional[str]): Optional prefix that nests the arguments.

        Returns:
            None
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        if "optim" not in skip:
            OF.add_class_args(parser, prefix="optim")
        if "lrsched" not in skip:
            LRSF.add_class_args(parser, prefix="lrsched")
        if "wdsched" not in skip:
            WDSF.add_class_args(parser, prefix="wdsched")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_train_modes_args(
        parser: ArgumentParser,
        train_modes: Optional[List[str]] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds the ``--train-mode`` CLI option with optional enum constraints.

        Args:
            parser (ArgumentParser): Parser receiving the argument.
            train_modes (Optional[List[str]]): Allowed train-mode names.

        Returns:
            None
        """
        if skip is None:
            skip = set()

        if train_modes is not None and "train_mode" not in skip:
            parser.add_argument(
                "--train-mode",
                default="full",
                choices=train_modes,
                help=f"Named train-mode to activate inside the model ({', '.join(train_modes)}).",
            )

    @staticmethod
    def add_io_keys_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds CLI options that specify the dataloader keys for inputs/targets.

        Args:
            parser (ArgumentParser): Parser receiving the arguments.
            prefix (Optional[str]): Optional namespace prefix.

        Returns:
            None
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        if "input_key" not in skip:
            parser.add_argument(
                "--input-key",
                default="audio_aug",
                help="Batch dictionary key that contains the tensor fed to the model.",
            )
        if "target_key" not in skip:
            parser.add_argument(
                "--target-key",
                default="speaker",
                help="Batch dictionary key that contains the supervision targets.",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        train_modes: Optional[List[str]] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Registers the full set of trainer CLI arguments, optionally nested
        under a prefix.

        Args:
            parser (ArgumentParser): Parser receiving the arguments.
            prefix (Optional[str]): Optional prefix for grouped arguments.
            train_modes (Optional[List[str]]): Allowed model train modes.
            skip (Set[str]): Currently unused (kept for API parity).

        Returns:
            None
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        TorchTrainerBase.add_class_args(parser, skip=skip)
        SingleModelTrainer.add_optim_args(parser, skip=skip)
        SingleModelTrainer.add_io_keys_args(parser, skip=skip)
        SingleModelTrainer.add_train_modes_args(
            parser, train_modes=train_modes, skip=skip
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
