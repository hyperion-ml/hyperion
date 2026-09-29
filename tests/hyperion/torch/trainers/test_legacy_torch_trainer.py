"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from unittest.mock import Mock, call

from hyperion.torch.trainers.legacy_torch_trainer import LegacyTorchTrainer


def test_update_model_steps_weight_decay_scheduler_before_optimizer():
    trainer = LegacyTorchTrainer.__new__(LegacyTorchTrainer)
    trainer.lr_scheduler = None
    trainer.wd_scheduler = Mock()
    trainer.in_swa = False
    trainer.model = Mock()
    trainer.optimizer = Mock()
    trainer.grad_clip = 0
    trainer.grad_clip_norm = 2
    trainer.use_amp = False
    trainer.grad_scaler = None
    trainer.global_step = 0
    trainer._update_model_by_optim = Mock()

    calls = Mock()
    calls.attach_mock(trainer.wd_scheduler.on_opt_step, "step_wd")
    calls.attach_mock(trainer._update_model_by_optim, "step_optimizer")

    trainer.update_model()

    assert calls.mock_calls == [
        call.step_wd(),
        call.step_optimizer(
            trainer.model,
            trainer.optimizer,
            trainer.grad_clip,
            trainer.grad_clip_norm,
            trainer.use_amp,
            trainer.grad_scaler,
        ),
    ]
    assert trainer.global_step == 1
