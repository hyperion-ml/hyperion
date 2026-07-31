"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args


class DINOLoss(nn.Module):
    """DINO distillation loss.

    Attributes:
        num_classes: Number of teacher classes.
        student_temp: Student temperature.
        teacher_temp: Final teacher temperature.
        teacher_initial_temp: Starting teacher temperature.
        temp_warmup_epochs: Number of epochs for temperature warmup.
        center_momentum: Momentum used when updating the teacher center.
        cur_teacher_temp: Current teacher temperature.
        center: Exponential moving average center for teacher outputs.
    """

    def __init__(
        self,
        num_classes: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        teacher_initial_temp: float = 0.04,
        temp_warmup_epochs: int = 30,
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the DINO loss.

        Args:
            num_classes: Number of DINO classes.
            student_temp: Temperature of the student distribution.
            teacher_temp: Final teacher temperature.
            teacher_initial_temp: Initial teacher temperature.
            temp_warmup_epochs: Number of warmup epochs for the teacher temperature.
            center_momentum: Momentum for the teacher center update.
        """
        super().__init__()
        self.num_classes = num_classes
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_initial_temp = teacher_initial_temp
        self.temp_warmup_epochs = temp_warmup_epochs
        self.center_momentum = center_momentum
        self.cur_teacher_temp = teacher_initial_temp
        self.register_buffer("center", torch.zeros(1, num_classes))

    def update_temp(self, epoch: int) -> None:
        """Updates the teacher temperature schedule.

        Args:
            epoch: Current epoch.
        """
        if epoch < self.temp_warmup_epochs:
            self.cur_teacher_temp = (
                self.teacher_initial_temp
                + (self.teacher_temp - self.teacher_initial_temp)
                * epoch
                / self.temp_warmup_epochs
            )
            logging.info("updating dino-loss teacher temp=%.3f", self.cur_teacher_temp)
        else:
            self.cur_teacher_temp = self.teacher_temp

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        num_student_crops: int,
        num_teacher_crops: int,
    ) -> torch.Tensor:
        """Computes the DINO cross-entropy between student and teacher outputs.

        Args:
            student_pred: Student logits.
            teacher_pred: Teacher logits.
            num_student_crops: Number of student crops in the batch.
            num_teacher_crops: Number of teacher crops in the batch.

        Returns:
            Scalar loss tensor.
        """
        # assert not torch.any(torch.isnan(student_pred)), f"loss/student is nan"
        student_pred = student_pred / self.student_temp
        # assert not torch.any(torch.isnan(student_pred)), f"loss/p is nan"
        student_pred = student_pred.chunk(num_student_crops)
        teacher_pred = teacher_pred.detach()
        center = self.center  # we take the center before updating it
        if self.training:
            self.update_center(teacher_pred)
        # assert not torch.any(torch.isnan(teacher_pred)), f"loss/teacher is nan"
        teacher_pred = nn.functional.softmax(
            (teacher_pred - center) / self.cur_teacher_temp, dim=-1
        )
        # assert not torch.any(torch.isnan(teacher_pred)), f"loss/q is nan {center}"
        teacher_pred = teacher_pred.chunk(num_teacher_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_pred):
            for ip, p in enumerate(student_pred):
                if ip == iq and num_teacher_crops > 1:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * nn.functional.log_softmax(p, dim=-1), dim=-1)
                # assert not torch.any(
                #     torch.isnan(loss)
                # ), f"loss is nan {iq} {ip} {torch.mean(q)} {torch.mean(p)} {torch.mean(center)}"
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_pred: torch.Tensor) -> None:
        """Updates the teacher output center.

        Args:
            teacher_pred: Teacher logits used to update the center.
        """
        batch_acc = torch.sum(teacher_pred, dim=0, keepdim=True)
        batch_size = torch.as_tensor(teacher_pred.size(0), device=batch_acc.device)
        if dist.is_initialized():
            dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_acc, op=dist.ReduceOp.SUM)

        batch_center = batch_acc / batch_size
        if torch.any(torch.isnan(batch_center)):
            logging.warning(f"batch-center is nan")
            return

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    @staticmethod
    def filter_args(**kwargs: object) -> dict:
        """Filters keyword arguments accepted by ``__init__``.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dictionary containing the accepted keyword arguments.
        """
        return filter_func_args(DINOLoss.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: str | None = None) -> None:
        """Adds CLI arguments for this loss.

        Args:
            parser: Argument parser to extend.
            prefix: Optional nested prefix for grouped arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--num-classes", default=65536, type=int, help="number of DINO classes"
        )
        parser.add_argument(
            "--student-temp",
            default=0.1,
            type=float,
            help="temperature of student distribution",
        )
        parser.add_argument(
            "--teacher-temp",
            default=0.07,
            type=float,
            help="final temperature of teacher distribution",
        )
        parser.add_argument(
            "--teacher-initial-temp",
            default=0.04,
            type=float,
            help="initial temperature of teacher distribution",
        )
        parser.add_argument(
            "--temp-warmup-epochs",
            default=30,
            type=int,
            help="warmup epochs for the teacher temperature",
        )
        parser.add_argument(
            "--center-momentum",
            default=0.9,
            type=float,
            help="momumntum for centering of the teacher distribution",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


class CosineDINOLoss(nn.Module):
    """Cosine regularizer for DINO embeddings.

    Attributes:
        scale: Target scale applied after warmup.
        warmup_epochs: Number of epochs for scale warmup.
        cur_scale: Current scale value.
    """

    def __init__(
        self,
        scale: float = 1.0,
        warmup_epochs: int = 30,
    ) -> None:
        """Initializes the cosine regularizer.

        Args:
            scale: Final scale value after warmup.
            warmup_epochs: Number of epochs for scale warmup.
        """
        super().__init__()
        self.scale = scale
        self.warmup_epochs = warmup_epochs
        self.cur_scale = scale

    def update_scale(self, epoch: int) -> None:
        """Updates the scale schedule.

        Args:
            epoch: Current epoch.
        """
        if epoch < self.warmup_epochs:
            self.cur_scale = self.scale * epoch / self.warmup_epochs
            logging.info("updating cosine-loss scale=%.3f", self.cur_scale)
        else:
            self.cur_scale = self.scale

    def forward(
        self,
        student_embed: torch.Tensor,
        teacher_embed: torch.Tensor,
        num_student_crops: int,
        num_teacher_crops: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the cosine regularization loss.

        Args:
            student_embed: Student embeddings.
            teacher_embed: Teacher embeddings.
            num_student_crops: Number of student crops in the batch.
            num_teacher_crops: Number of teacher crops in the batch.

        Returns:
            Tuple containing the scaled loss and the unscaled loss.
        """
        if self.scale == 0:
            zero = student_embed.new_zeros(())
            return zero, zero

        student_embed = torch.nn.functional.normalize(student_embed, dim=-1)
        teacher_embed = torch.nn.functional.normalize(teacher_embed, dim=-1)
        student_embed = student_embed.chunk(num_student_crops)
        teacher_embed = teacher_embed.detach()
        teacher_embed = teacher_embed.chunk(num_teacher_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_embed):
            for ip, p in enumerate(student_embed):
                if ip == iq and num_teacher_crops > 1:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = 1 - torch.sum(q * p, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        return self.cur_scale * total_loss, total_loss

    @staticmethod
    def filter_args(**kwargs: object) -> dict:
        """Filters keyword arguments accepted by ``__init__``.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dictionary containing the accepted keyword arguments.
        """
        return filter_func_args(CosineDINOLoss.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: str | None = None) -> None:
        """Adds CLI arguments for this loss.

        Args:
            parser: Argument parser to extend.
            prefix: Optional nested prefix for grouped arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--scale", default=0, type=float, help="Scale of Cosine loss to reg. DINO"
        )
        parser.add_argument(
            "--warmup-epochs",
            default=30,
            type=int,
            help="warmup epochs for the scale",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
