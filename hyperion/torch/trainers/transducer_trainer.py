"""
Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
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
from ..lr_schedulers import LRScheduler as LRS
from ..wd_schedulers import WDScheduler as WDS
from .single_model_trainer import SingleModelTrainer
from .torch_trainer_base import AMPDType, DDPType, FSDPMPDType


class TransducerTrainer(SingleModelTrainer):
    """Trainer for ASR transducer-style single-model training.

    Attributes:
        model (HyperTorchModel): Transducer model instance to optimize.
        optim (torch.optim.Optimizer | Dict[str, Any]): Optimizer or config dict.
        lrsched (Optional[LRS]): Learning-rate scheduler object or config dict.
        wdsched (Optional[WDS]): Weight-decay scheduler object or config dict.
        train_mode (str): Named train-mode activated inside the model.
        loss (Optional[nn.Module]): Optional external loss, retained for config parity.
        metrics (Dict[str, Any]): Extra metric callables applied to model output.
        input_key (str): Batch key containing acoustic features.
        target_key (str): Batch key containing text/token supervision.
        compile_model (bool): Enables ``torch.compile`` for the model forward.
        compile_dynamic (bool): Enables dynamic-shape compilation when compiling.
    """

    def __init__(
        self,
        model: HyperTorchModel,
        optim: Union[torch.optim.Optimizer, Dict[str, Any]],
        lrsched: Optional[Union[LRS, Dict[str, Any]]] = None,
        wdsched: Optional[Union[WDS, Dict[str, Any]]] = None,
        train_mode: str = "full",
        loss: Optional[nn.Module] = None,
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
        target_key: str = "text",
    ) -> None:
        """Initializes the transducer trainer.

        Args:
            model: Transducer model instance.
            optim: Optimizer instance or optimizer configuration dictionary.
            lrsched: Learning-rate scheduler instance or configuration dictionary.
            wdsched: Weight-decay scheduler instance or configuration dictionary.
            train_mode: Named train-mode to activate inside the model.
            loss: Optional external loss module retained for compatibility.
            metrics: Extra metric callables applied to model output and targets.
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
            input_key: Batch key containing acoustic features.
            target_key: Batch key containing text/token supervision.

        Returns:
            None.
        """
        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)
        self.metrics = {} if metrics is None else metrics

    def preprocess_data(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Extracts transducer inputs, input lengths, and targets from a batch.

        Args:
            batch_data: Batch emitted by the dataloader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch dictionary.
        """
        input_lengths_key = f"{self.input_key}_lengths"
        output_batch_data = {
            "audio": batch_data[self.input_key],
            "audio_lengths": batch_data[input_lengths_key],
            "target": batch_data[self.target_key],
        }
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(
        self, batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Any]:
        """Runs the transducer forward pass and returns the model loss.

        Args:
            batch_data: Preprocessed batch with ``audio``, ``audio_lengths``, and
                ``target`` tensors.

        Returns:
            Tuple[torch.Tensor, Any]: Scalar model loss and raw model output.
        """
        target = batch_data["target"]
        if hasattr(target, "to"):
            target = target.to(batch_data["audio"].device)
            batch_data["target"] = target

        output = self.model(
            batch_data["audio"],
            x_lengths=batch_data["audio_lengths"],
            y=target,
        )
        loss = output.loss.mean()
        return loss, output

    def compute_metrics(
        self, batch_output: Any, batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Computes loss-valued output metrics and optional external metrics.

        Args:
            batch_output: Model output object containing loss fields.
            batch_data: Preprocessed batch with ``target`` tensor.

        Returns:
            Dict[str, Any]: Metrics to aggregate for the current batch.
        """
        batch_metrics = ODict()
        for key, value in batch_output.items():
            if "loss" in key and value is not None:
                batch_metrics[key] = value.mean().item()

        target = batch_data["target"]
        for key, metric in self.metrics.items():
            batch_metrics[key] = metric(batch_output, target)

        return batch_metrics

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword arguments down to those accepted by ``__init__``.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Subset compatible with ``TransducerTrainer.__init__``.
        """
        return filter_func_args(TransducerTrainer.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        train_modes: Optional[List[str]] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Registers transducer trainer CLI arguments.

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
                help="Batch dictionary key that contains acoustic features.",
            )
        if "target_key" not in skip:
            parser.add_argument(
                "--target-key",
                default="text",
                help="Batch dictionary key that contains text/token targets.",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
