"""Hydra head prototype code-rate configuration coverage."""

from jsonargparse import ArgumentParser, namespace_to_dict

from hyperion.torch.narchs.hydra_heads import HydraClassifHead


def test_nested_prototype_code_rate_parser_and_config():
    parser = ArgumentParser()
    HydraClassifHead.add_class_args(parser)
    cfg = namespace_to_dict(
        parser.parse_args(
            [
                "--loss-type=softmax",
                "--num-classes=5",
                "--enable-prototype-code-rate",
                "--prototype_code_rate.eps=0.25",
                "--prototype_code_rate.jitter=0.0002",
                "--prototype_code_rate.gamma-1=1.5",
                "--prototype_code_rate.normalize",
            ]
        )
    )

    head = HydraClassifHead(
        in_feats=4,
        **HydraClassifHead.filter_args(**cfg),
    )
    assert head.code_rate.eps == 0.25
    assert head.code_rate.jitter == 0.0002
    assert head.code_rate.gamma_1 == 1.5
    assert head.code_rate.normalize
    assert head.code_rate.distributed_mode == "local"
    assert head.get_config()["prototype_code_rate"] == cfg["prototype_code_rate"]


def test_prototype_code_rate_reconfiguration():
    head = HydraClassifHead(
        in_feats=4,
        num_classes=5,
        loss_type="softmax",
        enable_prototype_code_rate=True,
        prototype_code_rate={"eps": 0.4},
    )
    head.reconfig_or_create(
        in_feats=4,
        num_classes=5,
        loss_type="softmax",
        enable_prototype_code_rate=True,
        prototype_code_rate={"eps": 0.2, "gamma_2": 3.0},
    )
    assert head.code_rate.eps == 0.2
    assert head.code_rate.gamma_2 == 3.0
    assert head.code_rate.normalize is True
    assert head.code_rate.distributed_mode == "local"
    assert head.get_config()["prototype_code_rate"] == {
        "eps": 0.2,
        "gamma_2": 3.0,
    }


def test_prototype_sig_reg_parser_forward_and_reconfiguration():
    import torch

    from hyperion.torch.narchs.hydra_head_factory import HydraHeadFactory

    parser = ArgumentParser()
    HydraHeadFactory.add_class_args(parser)
    cfg = parser.parse_args(
        ["--enable-prototype-sig-reg", "--prototype_sig_reg.num-slices=8"]
    ).as_dict()
    assert cfg["prototype_sig_reg"]["distributed_mode"] == "local"
    for loss_type in ("softmax", "cos-softmax", "arc-softmax", "subcenter-arc-softmax"):
        head = HydraClassifHead(
            in_feats=4,
            num_classes=5,
            loss_type=loss_type,
            enable_prototype_sig_reg=True,
            prototype_sig_reg={"num_slices": 8},
        )
        assert head.raw_prototypes.shape == (5, 4)
        output = head(torch.randn(2, 4))
        output.prototype_sig_reg.backward()
        assert torch.isfinite(output.prototype_sig_reg)
        parameter = head.output.weight if loss_type == "softmax" else head.output.kernel
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        clone = HydraClassifHead(**head.get_config(no_class_name=True))
        assert clone.sig_reg.distributed_mode == "local"
        head.reconfig_or_create(
            in_feats=4,
            num_classes=5,
            loss_type=loss_type,
            enable_prototype_sig_reg=True,
            prototype_sig_reg={"num_slices": 16},
        )
        assert head.sig_reg.num_slices == 16
        head.reconfig_or_create(in_feats=4, num_classes=5, loss_type=loss_type)
        assert not hasattr(head, "sig_reg")
        assert head(torch.randn(2, 4)).prototype_sig_reg is None


def test_raw_prototypes_and_internal_code_rate_normalization():
    import torch
    import torch.nn.functional as F

    from hyperion.torch.losses.rate_distortion import (
        SubspaceLikeGaussianCodeRateDistortionL2,
    )

    for loss_type in ("softmax", "cos-softmax", "arc-softmax", "subcenter-arc-softmax"):
        head = HydraClassifHead(
            in_feats=4,
            num_classes=5,
            loss_type=loss_type,
            enable_prototype_code_rate=True,
        )
        parameter = head.output.weight if loss_type == "softmax" else head.output.kernel
        with torch.no_grad():
            parameter.copy_(torch.arange(1, parameter.numel() + 1).view_as(parameter))
            if loss_type == "subcenter-arc-softmax":
                head.output.subcenter_counts[:, 1] = 10
        if loss_type == "softmax":
            expected = parameter
        elif loss_type == "subcenter-arc-softmax":
            expected = parameter.view(4, 5, 2)[:, :, 1].T
        else:
            expected = parameter.T
        torch.testing.assert_close(head.raw_prototypes, expected)
        if loss_type != "softmax":
            torch.testing.assert_close(head.output.raw_prototypes, expected)
            torch.testing.assert_close(
                head.output.prototypes, F.normalize(expected, dim=-1)
            )
        output = head(torch.randn(2, 4))
        reference = SubspaceLikeGaussianCodeRateDistortionL2(
            normalize=False, distributed_mode="local"
        )(F.normalize(expected, dim=-1))
        torch.testing.assert_close(output.prototype_code_rate, reference)
        output.prototype_code_rate.backward()
        assert torch.isfinite(parameter.grad).all()
        if loss_type == "subcenter-arc-softmax":
            assert torch.count_nonzero(parameter.grad.view(4, 5, 2)[:, :, 0]) == 0


def test_prototype_regularizers_match_explicit_values_and_gradients():
    from unittest.mock import patch

    import torch
    import torch.nn.functional as F

    for loss_type in ("softmax", "cos-softmax", "arc-softmax", "subcenter-arc-softmax"):
        head = HydraClassifHead(
            in_feats=4,
            num_classes=5,
            loss_type=loss_type,
            enable_prototype_code_rate=True,
            enable_prototype_sig_reg=True,
            prototype_sig_reg={"num_slices": 8},
        )
        parameter = head.output.weight if loss_type == "softmax" else head.output.kernel
        with torch.no_grad():
            parameter.copy_(torch.linspace(-2, 3, parameter.numel()).view_as(parameter))
            if loss_type == "subcenter-arc-softmax":
                head.output.subcenter_counts[:, 1] = 100
        directions = F.normalize(
            torch.arange(1, 33, dtype=torch.float32).view(4, 8), dim=0
        )
        with patch.object(head.sig_reg, "_sample_directions", return_value=directions):
            output = head(torch.randn(2, 4))
        raw = head.raw_prototypes.detach().clone().requires_grad_()
        normalized = F.normalize(raw, dim=-1)
        gram = normalized.T @ normalized
        rate_reference = torch.linalg.slogdet(
            torch.eye(4) * (1 + head.code_rate.jitter)
            + 4 / (5 * head.code_rate.eps**2) * gram
        )[1] / (2 * torch.log(torch.tensor(2.0)))
        t = torch.linspace(-5, 5, 17)
        target = torch.exp(-t.square() / 2)
        phase = (raw @ directions).unsqueeze(-1) * t
        error = (phase.cos().mean(0) - target).square() + phase.sin().mean(0).square()
        sig_reference = 5 * torch.trapz(error * target, t, dim=-1).mean()
        torch.testing.assert_close(output.prototype_code_rate, rate_reference)
        torch.testing.assert_close(output.prototype_sig_reg, sig_reference)
        for actual, reference in (
            (output.prototype_code_rate, rate_reference),
            (output.prototype_sig_reg, sig_reference),
        ):
            actual_grad = torch.autograd.grad(actual, parameter, retain_graph=True)[0]
            reference_grad = torch.autograd.grad(reference, raw, retain_graph=True)[0]
            if loss_type == "softmax":
                torch.testing.assert_close(actual_grad, reference_grad)
            elif loss_type == "subcenter-arc-softmax":
                torch.testing.assert_close(
                    actual_grad.view(4, 5, 2)[:, :, 1].T, reference_grad
                )
                assert torch.count_nonzero(actual_grad.view(4, 5, 2)[:, :, 0]) == 0
            else:
                torch.testing.assert_close(actual_grad.T, reference_grad)


def test_prototype_regularizers_force_local_statistics_and_normalization():
    """Prototype roles override conflicting distribution and normalization settings."""
    head = HydraClassifHead(
        in_feats=4,
        num_classes=5,
        enable_prototype_code_rate=True,
        prototype_code_rate={"distributed_mode": "global_data", "normalize": False},
        enable_prototype_sig_reg=True,
        prototype_sig_reg={"distributed_mode": "global_data"},
    )
    assert head.code_rate.distributed_mode == "local"
    assert head.code_rate.normalize is True
    assert head.sig_reg.distributed_mode == "local"


def test_reconfigured_prototype_sig_reg_uses_head_device():
    """Reconfiguration creates SIGReg buffers on the existing head device."""
    head = HydraClassifHead(in_feats=4, num_classes=5, loss_type="softmax").to("meta")
    head.reconfig_or_create(
        in_feats=4, num_classes=5, loss_type="softmax", enable_prototype_sig_reg=True
    )
    assert head.sig_reg.t.device == head.output.weight.device
    assert head.sig_reg.counter.device == head.output.weight.device
