"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..hyper_torch_model import HyperTorchModel
from ..loggers import LoggerList
from ..losses import BCEWithLLR
from ..lr_schedulers import LRScheduler as LRS
from ..utils.misc import get_selfsim_tarnon
from ..wd_schedulers import WDScheduler as WDS
from .single_model_trainer import SingleModelTrainer
from .torch_trainer_base import AMPDType, DDPType, FSDPMPDType


class PLDATrainer(SingleModelTrainer):
    """Trainer for PLDA back-end models.

    Attributes:
        model (HyperTorchModel): PLDA model instance to optimize.
        optim (torch.optim.Optimizer | Dict[str, Any]): Optimizer or config dict.
        lrsched (Optional[LRS]): Learning-rate scheduler object or config dict.
        wdsched (Optional[WDS]): Weight-decay scheduler object or config dict.
        train_mode (str): Named train-mode activated inside the model.
        loss (nn.Module): Multiclass loss applied to the ``multi`` output.
        loss_bce (BCEWithLLR): Binary target/non-target PLDA loss.
        muliclass_loss_weight (float): Weight for the multiclass PLDA loss.
        binary_loss_weight (float): Weight for the binary target/non-target PLDA loss.
        metrics (Dict[str, Any]): Extra metric callables applied to ``multi`` output.
        p_tar (float): Target prior used by the binary BCE-with-LLR criterion.
        input_key (str): Batch key containing PLDA inputs.
        target_key (str): Batch key containing class labels.
        compile_model (bool): Enables ``torch.compile`` for the model forward.
        compile_dynamic (bool): Enables dynamic-shape compilation when compiling.
    """

    def __init__(
        self,
        model: HyperTorchModel,
        optim: Union[torch.optim.Optimizer, Dict[str, Any]],
        lrsched: Optional[Union[LRS, Dict[str, Any]]] = None,
        wdsched: Optional[Union[WDS, Dict[str, Any]]] = None,
        train_mode: str = "train",
        loss: Optional[nn.Module] = None,
        muliclass_loss_weight: float = 1.0,
        binary_loss_weight: float = 0.0,
        p_tar: float = 0.5,
        metrics: Optional[Dict[str, Any]] = None,
        exp_path: PathLike = "./train",
        num_epochs: int = 100,
        cur_epoch: int = 0,
        max_steps: Optional[int] = None,
        cur_step: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        val_steps: Optional[int] = None,
        val_hours: Optional[float] = None,
        save_steps: Optional[int] = None,
        save_hours: Optional[float] = None,
        device: Union[torch.device, int, None] = None,
        loggers: Optional[LoggerList] = None,
        ddp: bool = False,
        ddp_type: DDPType = DDPType.DDP,
        fsdp_reshard_after_forward: Optional[Union[bool, int]] = None,
        fsdp_mp_param_dtype: Optional[FSDPMPDType] = None,
        fsdp_mp_reduce_dtype: Optional[FSDPMPDType] = None,
        fsdp_mp_output_dtype: Optional[FSDPMPDType] = None,
        fsdp_cpu_offload: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        bf16_grad_scaler: bool = False,
        log_interval: int = 1000,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Optional[Dict[str, str]] = None,
        grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: float = 1e-3,
        swa_update_steps: int = 50000,
        swa_anneal_steps: int = 50000,
        bn_update_steps: int = 5000,
        compile_model: bool = False,
        compile_dynamic: bool = False,
        input_key: str = "x",
        target_key: str = "class_id",
    ) -> None:
        """Initializes the PLDA trainer.

        Args:
            model: PLDA model instance.
            optim: Optimizer instance or optimizer configuration dictionary.
            lrsched: Learning-rate scheduler instance or configuration dictionary.
            wdsched: Weight-decay scheduler instance or configuration dictionary.
            train_mode: Named train-mode to activate inside the model.
            loss: Multiclass criterion for the ``multi`` output.
            muliclass_loss_weight: Weight for the multiclass PLDA loss.
            binary_loss_weight: Weight for the binary target/non-target PLDA loss.
            p_tar: Target prior used by ``BCEWithLLR``.
            metrics: Extra metric callables applied to ``multi`` output and targets.
            exp_path: Directory used to save checkpoints and logs.
            num_epochs: Maximum number of epochs to run.
            cur_epoch: Epoch index to resume from.
            max_steps: Optional cap on total optimizer steps.
            cur_step: Global optimizer step to resume from.
            grad_acc_steps: Gradient accumulation steps per optimizer update.
            eff_batch_size: Reference effective batch size.
            val_steps: Number of optimizer steps between validations.
            val_hours: Max wall-clock hours between validations.
            save_steps: Number of optimizer steps between checkpoint saves.
            save_hours: Max wall-clock hours between checkpoint saves.
            device: Device on which to run the model.
            loggers: Logger collection receiving progress events.
            ddp: Whether distributed training is enabled.
            ddp_type: Distributed backend type.
            fsdp_reshard_after_forward: FSDP2 reshard policy after forward.
            fsdp_mp_param_dtype: FSDP2 mixed-precision parameter dtype.
            fsdp_mp_reduce_dtype: FSDP2 mixed-precision reduction dtype.
            fsdp_mp_output_dtype: FSDP2 mixed-precision output dtype.
            fsdp_cpu_offload: Enables CPU offload for FSDP2.
            use_amp: Enables automatic mixed precision.
            amp_dtype: AMP precision used when ``use_amp`` is true.
            bf16_grad_scaler: Enables gradient scaling for bfloat16 AMP.
            log_interval: Optimizer steps between progress logs.
            use_tensorboard: Enables TensorBoard logging.
            use_wandb: Enables Weights & Biases logging.
            wandb: Additional W&B configuration.
            grad_clip: Gradient clipping threshold; non-positive disables clipping.
            grad_clip_norm: Norm type used for gradient clipping.
            swa_start: Step at which SWA averaging starts; 0 disables SWA.
            swa_lr: Learning rate used during SWA.
            swa_update_steps: Steps between SWA parameter averaging operations.
            swa_anneal_steps: Steps used to anneal the SWA learning rate.
            bn_update_steps: Max steps for SWA BatchNorm-statistics refresh.
            compile_model: Enables ``torch.compile`` for the model forward.
            compile_dynamic: Enables dynamic-shape compilation when compiling.
            input_key: Batch key containing PLDA inputs.
            target_key: Batch key containing class labels.

        Returns:
            None.
        """
        if loss is None:
            loss = nn.CrossEntropyLoss()
        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.loss_bce = BCEWithLLR(p_tar)
        self.muliclass_loss_weight = muliclass_loss_weight
        self.binary_loss_weight = binary_loss_weight
        self.metrics = {} if metrics is None else metrics
        self.p_tar = p_tar

    def on_epoch_begin(self) -> None:
        """Updates model epoch-dependent margins before each training epoch.

        Args:
            None.

        Returns:
            None.
        """
        super().on_epoch_begin()
        if hasattr(self.model, "update_margin"):
            self.model.update_margin(self.cur_epoch)

    def preprocess_data(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Extracts PLDA inputs and targets from a raw dataloader batch.

        Args:
            batch_data: Batch emitted by the dataloader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch dictionary.
        """
        output_batch_data = {
            "audio": batch_data[self.input_key],
            "target": batch_data[self.target_key],
        }
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(
        self, batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs the PLDA forward pass and computes configured losses.

        Args:
            batch_data: Preprocessed batch with ``audio`` and ``target`` tensors.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Scalar loss and model outputs.
        """
        return_multi = self.muliclass_loss_weight > 0
        return_bin = self.binary_loss_weight > 0
        if not return_multi and not return_bin:
            raise ValueError("At least one PLDA loss weight must be greater than zero.")

        input_data = batch_data["audio"]
        target = batch_data["target"]
        target_bin = None
        mask_bin = None
        if return_bin:
            target_bin, mask_bin = get_selfsim_tarnon(target, return_mask=True)

        if self.model.training:
            output = self.model(
                input_data,
                target,
                return_multi=return_multi,
                return_bin=return_bin,
                y_bin=target_bin,
            )
        else:
            output = self.model(
                input_data,
                return_multi=return_multi,
                return_bin=return_bin,
            )

        loss = input_data.new_zeros(())
        loss_multi = None
        loss_bin = None
        if return_multi:
            loss_multi = self.loss(output["multi"], target).mean()
            loss = loss + self.muliclass_loss_weight * loss_multi
        if return_bin:
            output_bin = output["bin"][mask_bin]
            target_bin = target_bin[mask_bin]
            loss_bin = self.loss_bce(output_bin, target_bin).mean()
            loss = loss + self.binary_loss_weight * loss_bin

        output["_loss_multi"] = loss_multi
        output["_loss_bin"] = loss_bin
        return loss, output

    def compute_metrics(
        self, batch_output: Dict[str, Any], batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Computes PLDA loss components and optional metrics for logging.

        Args:
            batch_output: Model outputs plus cached loss components.
            batch_data: Preprocessed batch with ``target`` tensor.

        Returns:
            Dict[str, Any]: Metrics to aggregate for the current batch.
        """
        batch_metrics = ODict()
        loss_bin = batch_output.get("_loss_bin")
        loss_multi = batch_output.get("_loss_multi")
        if loss_bin is not None:
            batch_metrics["loss_bin"] = loss_bin.item()
        if loss_multi is not None:
            target = batch_data["target"]
            batch_metrics["loss_multi"] = loss_multi.item()
            for key, metric in self.metrics.items():
                batch_metrics[key] = metric(batch_output["multi"], target)

        return batch_metrics

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword arguments down to those accepted by ``__init__``.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Subset compatible with ``PLDATrainer.__init__``.
        """
        return filter_func_args(PLDATrainer.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        train_modes: Optional[List[str]] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Registers PLDA trainer CLI arguments.

        Args:
            parser: Parser receiving the trainer arguments.
            prefix: Optional namespace prefix for grouped trainer arguments.
            train_modes: Optional list of allowed train modes.
            skip: Optional set of argument names to omit.

        Returns:
            None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        super_skip = skip.copy()
        super_skip.update({"input_key", "target_key"})
        SingleModelTrainer.add_class_args(
            parser, train_modes=train_modes, skip=super_skip
        )

        if "input_key" not in skip:
            parser.add_argument(
                "--input-key",
                default="x",
                help="Batch dictionary key that contains PLDA inputs.",
            )
        if "target_key" not in skip:
            parser.add_argument(
                "--target-key",
                default="class_id",
                help="Batch dictionary key that contains PLDA class labels.",
            )

        if "muliclass_loss_weight" not in skip:
            parser.add_argument(
                "--muliclass-loss-weight",
                default=1.0,
                type=float,
                help="Weight for the multiclass PLDA loss.",
            )
        if "binary_loss_weight" not in skip:
            parser.add_argument(
                "--binary-loss-weight",
                default=0.0,
                type=float,
                help="Weight for the binary target/non-target PLDA loss.",
            )
        if "p_tar" not in skip:
            parser.add_argument(
                "--p-tar",
                default=0.5,
                type=float,
                help="Target prior for the binary BCE-with-LLR loss.",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
