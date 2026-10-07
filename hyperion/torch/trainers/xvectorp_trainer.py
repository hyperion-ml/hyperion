"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict as ODict
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..hyper_torch_model import HyperTorchModel
from ..loggers import LoggerList
from ..lr_schedulers import LRScheduler as LRS
from ..metrics import CategoricalAccuracy
from ..models.xvectorps import XVectorPTrainMode
from ..narchs.hydra_heads import HydraClassifHeadOutput, HydraRegressionHeadOutput
from ..wd_schedulers import WDScheduler as WDS
from .single_model_trainer import SingleModelTrainer
from .torch_trainer_base import AMPDType, DDPType, FSDPMPDType, TorchTrainerBase

# from torch.distributed.elastic.multiprocessing.errors import record


class XVectorPTrainer(SingleModelTrainer):
    """Trainer specialized for X-vector+ models with categorical accuracy and prototype code-rate tracking.

    Attributes (including inherited members):

      model (HyperTorchModel): Model instance to optimize.
      optim (torch.optim.Optimizer or Dict[str, Any]): Optimizer or its configuration.
      lrsched (Optional[LRS or Dict[str, Any]]): Learning-rate scheduler or its
        configuration.
      wdsched (Optional[WDS or Dict[str, Any]]): Weight-decay scheduler or its
        configuration.
      train_mode (str): Name of the model training mode to activate, for example ``"full"``.
      exp_path (PathLike): Directory for checkpoints and logs.
      num_epochs (int): Total number of epochs to run.
      cur_epoch (int): Epoch index from which to resume.
      max_steps (Optional[int]): Global step budget overriding the epoch count.
      cur_step (int): Current global optimization step.
      grad_acc_steps (int): Minibatches accumulated before each optimizer step.
      eff_batch_size (Optional[int]): Reference effective batch size.
      val_steps (Optional[int]): Steps between validation passes.
      val_hours (Optional[float]): Wall-clock hours between validation passes.
      save_steps (Optional[int]): Steps between checkpoint saves.
      save_hours (Optional[float]): Wall-clock hours between checkpoint saves.
      device (torch.device, int, or None): Device on which the model executes.
      loggers (LoggerList): Active logger instances.
      ddp (bool): Whether DistributedDataParallel is enabled.
      ddp_type (DDPType): Selected distributed-data-parallel backend flavor.
      fsdp_reshard_after_forward (bool, int, or None): FSDP2 reshard policy after the forward pass.
      fsdp_mp_param_dtype (FSDPMPDType or None): FSDP2 mixed-precision parameter dtype.
      fsdp_mp_reduce_dtype (FSDPMPDType or None): FSDP2 mixed-precision reduction dtype.
      fsdp_mp_output_dtype (FSDPMPDType or None): FSDP2 mixed-precision output dtype.
      fsdp_cpu_offload (bool): Enables CPU offload for FSDP2.
      use_amp (bool): Enables automatic mixed precision.
      amp_dtype (AMPDType): Precision (float16 or bfloat16) used with AMP.
      bf16_grad_scaler (bool): Enables GradScaler with bfloat16 AMP.
      log_interval (int): Step interval between progress logs.
      use_tensorboard (bool): Enables TensorBoard logging.
      use_wandb (bool): Enables Weights & Biases logging.
      wandb (Dict[str, Any]): Additional Weights & Biases configuration.
      grad_clip (float): Gradient-norm clipping threshold.
      grad_clip_norm (str or int): Norm definition used for clipping.
      swa_start (int): Step at which to begin stochastic weight averaging.
      swa_lr (float): Learning rate used during stochastic weight averaging.
      swa_anneal_steps (int): Steps used for SWA learning-rate annealing.
      swa_update_steps (int): Interval between SWA weight updates.
      bn_update_steps (int): Maximum steps used to refresh batch-norm statistics after SWA.
      compile_model (bool): Enables ``torch.compile`` for model forward passes.
      compile_dynamic (bool): Enables dynamic-shape compilation.
      input_key (str): Key for the audio tensor in dataloader batches.
      target_key (str): Key for supervision labels in dataloader batches.
      xvector_sig_reg_weight: Weight added for global x-vector SIGReg.
      prototype_sig_reg_weight (float): Weight added for prototype SIGReg.
      prototype_code_rate_weight (float): Weight applied to the prototype
        code-rate regularizer in the total loss.
      categorical_acc_metric (CategoricalAccuracy): Metric accumulator used
        when the model exposes ``HydraClassifHeadOutput``.
    """

    def __init__(
        self,
        model: HyperTorchModel,
        optim: Union[torch.optim.Optimizer, Dict[str, Any]],
        lrsched: Optional[Union[LRS, Dict[str, Any]]] = None,
        wdsched: Optional[Union[WDS, Dict[str, Any]]] = None,
        train_mode: str = "full",
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
        input_key: str = "audio",
        target_key: str = "speaker",
        prototype_code_rate_weight: float = 0.0,
        prototype_sig_reg_weight: float = 0.0,
        xvector_sig_reg_weight: float = 0.0,
    ) -> None:
        """
        Initializes the X-vector+ trainer, forwarding most configuration to
        :class:`SingleModelTrainer` while setting the default IO keys and
        attaching a categorical-accuracy metric and optional prototype code-rate loss.

        Args:
            model (HyperTorchModel): Model instance to optimize.
            optim (torch.optim.Optimizer or Dict[str, Any]): Optimizer instance
                or configuration.
            lrsched (Optional[LRS or Dict[str, Any]]): Scheduler instance or
                configuration.
            wdsched (Optional[WDS or Dict[str, Any]]): Scheduler instance or
                configuration.
            train_mode (str): Model train-mode to activate (see ``XVectorPTrainMode``).
            exp_path (PathLike): Directory for checkpoints/logs.
            num_epochs (int): Maximum number of epochs to run.
            cur_epoch (int): Epoch to resume from.
            max_steps (Optional[int]): Optional global-step cap.
            cur_step (int): Global step to resume from.
            grad_acc_steps (int): Gradient accumulation steps.
            eff_batch_size (Optional[int]): Reference effective batch size.
            val_steps (Optional[int]): Steps between validations.
            val_hours (Optional[float]): Wall-clock hours between validation passes.
            save_steps (Optional[int]): Steps between checkpoint saves.
            save_hours (Optional[float]): Wall-clock hours between checkpoint saves.
            device (Union[torch.device, int, None]): Device to train on.
            loggers (Optional[LoggerList]): Logger collection.
            ddp (bool): Enables DDP training when True.
            ddp_type (DDPType): DDP backend flavor.
            fsdp_reshard_after_forward (bool|int|None): FSDP2 reshard policy after forward.
            fsdp_mp_param_dtype (FSDPMPDType|None): FSDP2 mixed-precision param dtype.
            fsdp_mp_reduce_dtype (FSDPMPDType|None): FSDP2 mixed-precision reduce dtype.
            fsdp_mp_output_dtype (FSDPMPDType|None): FSDP2 mixed-precision output dtype.
            fsdp_cpu_offload (bool): Enables FSDP CPU offload.
            use_amp (bool): Enables automatic mixed precision.
            amp_dtype (AMPDType): Precision to use when AMP is enabled.
            bf16_grad_scaler (bool): Enables GradScaler when using bfloat16 AMP.
            log_interval (int): Steps between logger updates.
            use_tensorboard (bool): Enables TensorBoard logging.
            use_wandb (bool): Enables W&B logging.
            wandb (Dict[str, str]): Extra W&B init parameters.
            grad_clip (float): Gradient clipping threshold (<=0 disables).
            grad_clip_norm (Union[str, int]): Norm used for clipping.
            swa_start (int): Step at which to start SWA averaging.
            swa_lr (float): SWA learning rate.
            swa_update_steps (int): Steps between SWA weight updates.
            swa_anneal_steps (int): Steps to anneal the SWA LR.
            bn_update_steps (int): Steps used to refresh BatchNorm statistics after SWA.
            compile_model (bool): Enables ``torch.compile`` for the model forward.
            compile_dynamic (bool): Enables dynamic-shape compilation when compiling.
            input_key (str): Batch key used for the audio tensor.
            target_key (str): Batch key used for label tensors.
            prototype_sig_reg_weight (float): Weight added for prototype SIGReg.
            xvector_sig_reg_weight: Weight added for global x-vector SIGReg.
            prototype_code_rate_weight (float): Weight applied to the prototype
                code-rate regularizer.

        """

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.prototype_code_rate_weight = prototype_code_rate_weight
        self.prototype_sig_reg_weight = prototype_sig_reg_weight
        self.xvector_sig_reg_weight = xvector_sig_reg_weight
        self.categorical_acc_metric = CategoricalAccuracy()

    def preprocess_data(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Normalizes dataloader batches to the ``audio``/``target`` interface
        expected by :class:`SingleModelTrainer`, preserving optional audio lengths.

        Args:
            batch_data (Dict[str, Any]): Raw batch emitted by the XVectorP dataloader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and the processed batch dict.
        """
        x_lengths_key = f"{self.input_key}_lengths"
        output_batch_data = {
            "audio": batch_data[self.input_key],
            "target": batch_data[self.target_key],
        }
        if x_lengths_key in batch_data:
            output_batch_data["audio_lengths"] = batch_data[x_lengths_key]
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Runs the model forward pass and composes the total optimization loss from
        the head loss and optional code-rate and SIGReg regularizers.

        Args:
            batch_data (Dict[str, Any]): Preprocessed batch from ``preprocess_data``.

        Returns:
            Tuple[torch.Tensor, Any]: Loss tensor and structured model output.
        """
        self.model.update_hyperparams(self.cur_step)
        batch_output = self.model(**batch_data)
        head_output = batch_output.head_output
        if not isinstance(
            head_output, (HydraClassifHeadOutput, HydraRegressionHeadOutput)
        ):
            raise ValueError(
                "XVectorPTrainer requires a classification or regression head"
            )
        loss = head_output.loss
        if loss is None:
            raise ValueError("XVectorPTrainer requires a head that returns a loss")

        if (
            isinstance(head_output, HydraClassifHeadOutput)
            and head_output.prototype_code_rate is not None
            and self.prototype_code_rate_weight != 0
        ):
            loss = (
                loss - self.prototype_code_rate_weight * head_output.prototype_code_rate
            )

        if (
            isinstance(head_output, HydraClassifHeadOutput)
            and head_output.prototype_sig_reg is not None
            and self.prototype_sig_reg_weight != 0
        ):
            loss = loss + self.prototype_sig_reg_weight * head_output.prototype_sig_reg

        if (
            batch_output.xvector_sig_reg is not None
            and self.xvector_sig_reg_weight != 0
        ):
            loss = loss + self.xvector_sig_reg_weight * batch_output.xvector_sig_reg

        return loss, batch_output

    def compute_metrics(
        self, batch_output: Any, batch_data: Dict[str, Any]
    ) -> ODict[str, float]:
        """
        Computes per-batch metrics (categorical accuracy when supported by the
        model head) for logging.

        Args:
            batch_output (Any): Structured model output that includes ``head_output``.
            batch_data (Dict[str, Any]): Input batch (needed for ground-truth labels).

        Returns:
            OrderedDict: Metrics keyed by descriptive names (e.g., ``categorical_acc``).
        """
        batch_metrics = ODict()
        if isinstance(batch_output.head_output, HydraClassifHeadOutput):
            categorical_acc = self.categorical_acc_metric(
                batch_output.head_output.logits, batch_data["target"]
            )

            batch_metrics["categorical_acc"] = categorical_acc
            if batch_output.head_output.loss is not None:
                batch_metrics["classification_loss"] = (
                    batch_output.head_output.loss.item()
                )
            if batch_output.head_output.prototype_code_rate is not None:
                batch_metrics["prototype_code_rate"] = (
                    batch_output.head_output.prototype_code_rate.item()
                )
            if batch_output.head_output.prototype_sig_reg is not None:
                batch_metrics["prototype_sig_reg"] = (
                    batch_output.head_output.prototype_sig_reg.item()
                )
        elif isinstance(batch_output.head_output, HydraRegressionHeadOutput):
            if batch_output.head_output.loss is not None:
                batch_metrics["regression_loss"] = batch_output.head_output.loss.item()
        elif batch_output.head_output is not None:
            logging.warning(
                "XVectorPTrainer: compute_metrics: Unknown head_output type %s",
                type(batch_output.head_output),
            )

        if batch_output.xvector_sig_reg is not None:
            batch_metrics["xvector_sig_reg"] = (
                batch_output.xvector_sig_reg.detach().item()
            )
        return batch_metrics

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filters keyword arguments down to those accepted by ``__init__`` so
        configs can be safely forwarded.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Subset compatible with :class:`XVectorPTrainer`.
        """
        args = filter_func_args(XVectorPTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Registers CLI arguments required to construct a :class:`XVectorPTrainer`,
        reusing the helper builders defined on :class:`SingleModelTrainer`.

        Args:
            parser (ArgumentParser): Parser that will receive the arguments.
            prefix (Optional[str]): Optional namespace prefix (Hydra-style).
            skip (Optional[Set[str]]): Argument names to skip when registering
                trainer, optimizer, IO-key, and train-mode options.

        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        TorchTrainerBase.add_class_args(parser, skip=skip)
        SingleModelTrainer.add_optim_args(parser, skip=skip)
        SingleModelTrainer.add_io_keys_args(parser, skip=skip)
        train_modes = XVectorPTrainMode.choices()
        SingleModelTrainer.add_train_modes_args(
            parser, train_modes=train_modes, skip=skip
        )

        if "xvector_sig_reg_weight" not in skip:
            parser.add_argument(
                "--xvector-sig-reg-weight",
                type=float,
                default=0.0,
                help="Weight added for global x-vector SIGReg.",
            )

        if "prototype_sig_reg_weight" not in skip:
            parser.add_argument(
                "--prototype-sig-reg-weight",
                type=float,
                default=0.0,
                help="Weight added for the prototype SIGReg regularizer.",
            )

        if "prototype_code_rate_weight" not in skip:
            parser.add_argument(
                "--prototype-code-rate-weight",
                type=float,
                default=0.0,
                help="Weight applied to the prototype code-rate regularizer.",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
