"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..loggers import LoggerList
from ..lr_schedulers import LRScheduler as LRS
from ..metrics import CategoricalAccuracy
from ..models.qvectors import QVectorTrainMode
from ..narchs.hydra_heads import HydraClassifHeadOutput
from ..hyper_torch_model import HyperTorchModel
from ..wd_schedulers import WDScheduler as WDS
from .single_model_trainer import SingleModelTrainer
from .torch_trainer_base import AMPDType, DDPType, FSDPMPDType, TorchTrainerBase

# from torch.distributed.elastic.multiprocessing.errors import record


class QVectorTrainer(SingleModelTrainer):
    """Trainer specialized for Q-vector models with categorical accuracy tracking.

    Attributes (includes inherited members):
        model (HyperTorchModel): Model instance to optimize.
        optim (torch.optim.Optimizer | Dict[str, Any]): Optimizer or optimizer config.
        lrsched (Optional[LRS]): Learning-rate scheduler or configuration dict.
        wdsched (Optional[WDS]): Weight-decay scheduler or configuration dict.
        train_mode (str): Name of the model's train mode to activate (e.g., ``\"full\"``).
        exp_path (PathLike): Directory for checkpoints/logs.
        num_epochs (int): Total training epochs to run.
        cur_epoch (int): Epoch index to resume from.
        max_steps (Optional[int]): Global step budget overriding epoch count.
        cur_step (int): Current global optimization step.
        grad_acc_steps (int): Minibatches accumulated before each optimizer step.
        eff_batch_size (Optional[int]): Reference effective batch size.
        val_steps (Optional[int]): Steps between validation passes.
        val_hours (Optional[float]): Wall-clock hours between validation passes.
        save_steps (Optional[int]): Steps between checkpoint saves.
        save_hours (Optional[float]): Wall-clock hours between checkpoint saves.
        device (Union[torch.device, int, None]): Device where the model executes.
        loggers (LoggerList): Active logger instances.
        ddp (bool): Whether DistributedDataParallel is enabled.
        ddp_type (DDPType): Selected DDP backend flavor.
        fsdp_reshard_after_forward (bool|int|None): FSDP2 reshard policy after forward.
        fsdp_mp_param_dtype (FSDPMPDType|None): FSDP2 mixed-precision parameter dtype.
        fsdp_mp_reduce_dtype (FSDPMPDType|None): FSDP2 mixed-precision reduction dtype.
        fsdp_mp_output_dtype (FSDPMPDType|None): FSDP2 mixed-precision output dtype.
        fsdp_cpu_offload (bool): Enables CPU offload for FSDP2.
        use_amp (bool): Enables automatic mixed precision.
        amp_dtype (AMPDType): Precision (float16/bfloat16) to use with AMP.
        log_interval (int): Step interval between progress logs.
        use_tensorboard (bool): Enables TensorBoard logging.
        use_wandb (bool): Enables Weights & Biases logging.
        wandb (Dict[str, Any]): Additional W&B configuration.
        grad_clip (float): Gradient-norm clipping threshold.
        grad_clip_norm (Union[str, int]): Norm definition for clipping.
        swa_start (int): Step at which to begin SWA averaging.
        swa_lr (float): Learning rate used while performing SWA.
        swa_anneal_steps (int): Steps used for SWA LR annealing.
        swa_update_steps (int): Interval between SWA weight updates.
        bn_update_steps (int): Max steps for BN statistics refresh after SWA.
        input_key (str): Key for the audio tensor in dataloader batches.
        target_key (str): Key for supervision labels in the batch.
        categorical_acc_metric (CategoricalAccuracy): Metric accumulator used when
            the model exposes a `HydraClassifHeadOutput`.
    """

    def __init__(
        self,
        model: HyperTorchModel,
        optim: torch.optim.Optimizer,
        lrsched: Optional[LRS] = None,
        wdsched: Optional[WDS] = None,
        train_mode: str = "full",
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
        wandb: Dict[str, str] = {},
        grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        input_key: str = "audio",
        target_key: str = "speaker",
    ) -> None:
        """
        Initializes the Q-vector trainer, forwarding most configuration to
        :class:`SingleModelTrainer` while setting the default IO keys and
        attaching a categorical-accuracy metric.

        Args:
            model (HyperTorchModel): Model instance to optimize.
            optim (torch.optim.Optimizer): Optimizer already configured for the model.
            lrsched (Optional[LRS]): Learning-rate scheduler or config dict.
            wdsched (Optional[WDS]): Weight-decay scheduler or config dict.
            train_mode (str): Model train-mode to activate (see ``QVectorTrainMode``).
            exp_path (PathLike): Directory for checkpoints/logs.
            num_epochs (int): Maximum number of epochs to run.
            cur_epoch (int): Epoch to resume from.
            max_steps (Optional[int]): Optional global-step cap.
            cur_step (int): Global step to resume from.
            grad_acc_steps (int): Gradient accumulation steps.
            eff_batch_size (Optional[int]): Reference effective batch size.
            val_steps (Optional[int]): Steps between validations.
            save_steps (Optional[int]): Steps between checkpoint saves.
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
            log_interval (int): Steps between logger updates.
            use_tensorboard (bool): Enables TensorBoard logging.
            use_wandb (bool): Enables W&B logging.
            wandb (Dict[str, str]): Extra W&B init parameters.
            grad_clip (float): Gradient clipping threshold (<=0 disables).
            grad_clip_norm (Union[str, int]): Norm used for clipping.
            swa_start (int): Step at which to start SWA averaging.
            swa_lr (int): SWA learning rate.
            swa_anneal_steps (int): Steps to anneal the SWA LR.
            input_key (str): Batch key used for the audio tensor.
            target_key (str): Batch key used for label tensors.

        Returns:
            None
        """

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.categorical_acc_metric = CategoricalAccuracy()

    # def preprocess_train_data(self, batch_data):
    #     # we get the keys of all augmented versions, except the non augmented one
    #     aug_keys = self.get_augs_keys(
    #         batch_data, self.input_key, skip=set(self.input_key)
    #     )
    #     x_lengths = batch_data[f"{self.input_key}_length"]
    #     y = batch_data[self.target_key]
    #     if aug_keys:
    #         # we concatenate all augmentations
    #         xs = []
    #         for key in aug_keys:
    #             xs.append(batch_data[key])
    #         x = torch.cat(xs, dim=0)
    #         x_lengths = torch.cat(len(aug_keys) * [x_lengths], dim=0)
    #         y = torch.cat(len(aug_keys) * [y], dim=0)
    #     if not aug_keys:
    #         x = batch_data[self.input_key]

    #     batch_data = {"x": x, "x_lengths": x_lengths, "y": y}
    #     batch_size = batch_data["x"].size(0)
    #     return batch_size, batch_data

    def preprocess_data(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Normalizes dataloader batches so they match the input interface expected
        by :class:`SingleModelTrainer` (keys ``audio``/``target`` plus optional lengths).

        Args:
            batch_data (Dict[str, Any]): Raw batch emitted by the q-vector dataloader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and the processed batch dict.
        """
        x_lengths_key = f"{self.input_key}_lengths"
        # y_lengths_key = f"{self.target_key}_lengths"
        output_batch_data = {
            # "id": batch_data["id"],
            "audio": batch_data[self.input_key],
            "target": batch_data[self.target_key],
        }
        if x_lengths_key in batch_data:
            output_batch_data["audio_lengths"] = batch_data[x_lengths_key]
        # if y_lengths_key in batch_data:
        #     output_batch_data["target_lengths"] = batch_data[y_lengths_key]
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Runs the model forward pass and extracts the scalar loss from the
        hydra classification head.

        Args:
            batch_data (Dict[str, Any]): Preprocessed batch from ``preprocess_data``.

        Returns:
            Tuple[torch.Tensor, Any]: Loss tensor and structured model output.
        """
        self.model.update_hyperparams(self.cur_step)
        batch_output = self.model(**batch_data)
        loss = batch_output.head_output.loss
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
        else:
            logging.warning(
                "QVectorTrainer: compute_metrics: Unknown head_output type %s"
                % type(batch_output.head_output)
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
            Dict[str, Any]: Subset compatible with :class:`QVectorTrainer`.
        """
        args = filter_func_args(QVectorTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Registers CLI arguments required to construct a :class:`QVectorTrainer`,
        reusing the helper builders defined on :class:`SingleModelTrainer`.

        Args:
            parser (ArgumentParser): Parser that will receive the arguments.
            prefix (Optional[str]): Optional namespace prefix (Hydra-style).
            skip (Set[str]): Unused placeholder for API compatibility.

        Returns:
            None
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        TorchTrainerBase.add_class_args(parser, skip=skip)
        SingleModelTrainer.add_optim_args(parser, skip=skip)
        SingleModelTrainer.add_io_keys_args(parser, skip=skip)
        train_modes = QVectorTrainMode.choices()
        SingleModelTrainer.add_train_modes_args(
            parser, train_modes=train_modes, skip=skip
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
