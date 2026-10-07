"""Prototype regularizer composition in embedding trainers."""

from types import SimpleNamespace

import pytest
import torch
from jsonargparse import ArgumentParser

from hyperion.torch.narchs.hydra_heads import HydraClassifHeadOutput
from hyperion.torch.trainers.qvector_trainer import QVectorTrainer
from hyperion.torch.trainers.xvectorp_trainer import XVectorPTrainer


@pytest.mark.parametrize("trainer_class", [XVectorPTrainer, QVectorTrainer])
@pytest.mark.parametrize("sig_weight", [0.0, 0.25])
def test_prototype_sig_reg_composition(trainer_class, sig_weight):
    """Check loss signs, gradients, and logging for both trainers.

    Args:
        trainer_class: Trainer implementation to exercise.
        sig_weight: SIGReg weight, including the disabled default.
    """
    trainer = trainer_class.__new__(trainer_class)
    trainer.cur_step = 0
    trainer.prototype_code_rate_weight = 0.5
    trainer.prototype_sig_reg_weight = sig_weight
    trainer.xvector_sig_reg_weight = 0.0
    trainer.qmatrix_code_rate_weight = 0.0
    trainer.categorical_acc_metric = lambda logits, target: 1.0
    ce = torch.tensor(3.0, requires_grad=True)
    rate = torch.tensor(2.0, requires_grad=True)
    sig = torch.tensor(4.0, requires_grad=True)
    output = SimpleNamespace(
        head_output=HydraClassifHeadOutput(
            logits=torch.zeros(2, 3),
            loss=ce,
            prototype_code_rate=rate,
            prototype_sig_reg=sig,
        ),
        qmatrix_code_rate=None,
        xvector_sig_reg=None,
    )

    class Model:
        def update_hyperparams(self, step):
            pass

        def __call__(self, **kwargs):
            return output

    trainer.model = Model()
    loss, actual_output = trainer.compute_forward({})
    torch.testing.assert_close(loss, ce - 0.5 * rate + sig_weight * sig)
    loss.backward()
    assert ce.grad == 1
    assert rate.grad == -0.5
    assert sig.grad == sig_weight if sig_weight else sig.grad is None
    assert actual_output is output
    metrics = trainer.compute_metrics(
        output, {"target": torch.zeros(2, dtype=torch.long)}
    )
    assert metrics["prototype_sig_reg"] == 4.0
    output.head_output.prototype_sig_reg = None
    loss, _ = trainer.compute_forward({})
    torch.testing.assert_close(loss, ce - 0.5 * rate)
    assert "prototype_sig_reg" not in trainer.compute_metrics(
        output, {"target": torch.zeros(2, dtype=torch.long)}
    )
    parser = ArgumentParser()
    trainer_class.add_class_args(parser)
    assert parser.parse_args([]).prototype_sig_reg_weight == 0.0
    cfg = parser.parse_args(["--prototype-sig-reg-weight=0.25"]).as_dict()
    assert trainer_class.filter_args(**cfg)["prototype_sig_reg_weight"] == 0.25


def test_model_xvector_sig_reg_is_weighted_and_logged():
    """The trainer consumes a model-produced statistic without owning SIGReg."""
    trainer = XVectorPTrainer.__new__(XVectorPTrainer)
    trainer.cur_step = 0
    trainer.prototype_code_rate_weight = 0.0
    trainer.prototype_sig_reg_weight = 0.0
    trainer.xvector_sig_reg_weight = 0.25
    trainer.categorical_acc_metric = lambda logits, target: 1.0
    sig = torch.tensor(4.0, requires_grad=True)
    output = SimpleNamespace(
        xvector_sig_reg=sig,
        head_output=HydraClassifHeadOutput(
            logits=torch.zeros(2, 3), loss=torch.tensor(2.0)
        ),
    )

    class Model:
        def update_hyperparams(self, step):
            pass

        def __call__(self, **kwargs):
            return output

    trainer.model = Model()
    loss, _ = trainer.compute_forward({})
    assert loss.item() == 3.0
    loss.backward()
    assert sig.grad.item() == 0.25
    assert (
        trainer.compute_metrics(output, {"target": torch.zeros(2, dtype=torch.long)})[
            "xvector_sig_reg"
        ]
        == 4.0
    )
    output.xvector_sig_reg = None
    loss, _ = trainer.compute_forward({})
    assert loss.item() == 2.0
    assert "xvector_sig_reg" not in trainer.compute_metrics(
        output, {"target": torch.zeros(2, dtype=torch.long)}
    )
