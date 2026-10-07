"""Regression tests for global pooling and projection in x-vector+ models."""

import json

import pytest
import torch
from jsonargparse import ArgumentParser, namespace_to_dict
from torch import nn

from hyperion.torch.models.xvectorps import XVectorP, XVectorPOutput, XVectorPTrainMode
from hyperion.torch.narchs import ProjHead


class TinyXVectorP(XVectorP):
    """Small waveform backbone for testing the base model contract.

    Attributes:
        backbone: Frame-wise projection from samples to four features.
        pooling: Global pooling module (inherited).
        proj_head: Embedding projection (inherited).
        head: Downstream classification head (inherited).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = nn.Linear(1, 4)

    def backbone_output_feats(self):
        return 4

    @property
    def sample_frequency(self):
        return 10

    def forward_backbone(self, x, x_lengths=None, return_hidden_feats=False):
        feats = self.backbone(x.unsqueeze(-1))
        return feats, x_lengths, [feats] if return_hidden_feats else None, x_lengths

    def set_backbone_in_eval_mode(self):
        self.backbone.eval()

    def set_adapters_in_eval_mode(self):
        pass

    def freeze_backbone_feat_extractor(self):
        self.backbone.requires_grad_(False)

    def set_backbone_feat_extractor_in_eval_mode(self):
        self.backbone.eval()


def make_model(**kwargs):
    """Construct a tiny model with a softmax head.

    Args:
        **kwargs: Constructor overrides.

    Returns:
        Tiny x-vector+ model.
    """
    config = {
        "pooling": "mean+stddev",
        "xvector_dim": 3,
        "head": {"head_type": "classif", "num_classes": 2, "loss_type": "softmax"},
        "proj_use_norm": False,
    }
    config.update(kwargs)
    return TinyXVectorP(**config)


@pytest.mark.parametrize("norm", ["batch-norm", "layer-norm", "rms-norm"])
@pytest.mark.parametrize("before", [True, False])
def test_projection_normalization_order(norm, before):
    proj = ProjHead(4, 3, norm_layer=norm, norm_before=before).eval()
    assert proj.in_shape() == (None, 4)
    assert proj.out_shape((2, 4)) == (2, 3)
    x = torch.randn(2, 4)
    expected = (
        proj.proj(proj._norm_layer(x)) if before else proj._norm_layer(proj.proj(x))
    )
    torch.testing.assert_close(proj(x), expected)
    assert proj._norm_layer.weight.numel() == (4 if before else 3)
    cfg = proj.get_config(no_class_name=True)
    rebuilt = ProjHead(**cfg)
    rebuilt.load_state_dict(proj.state_dict())
    torch.testing.assert_close(rebuilt.eval()(x), expected)


@pytest.mark.parametrize("norm", ["batch-norm", "layer-norm", "rms-norm"])
@pytest.mark.parametrize("before", [True, False])
def test_model_bias_is_determined_by_normalization_order(norm, before):
    model = make_model(
        proj_norm_layer=norm, proj_use_norm=True, proj_norm_before=before
    )
    assert (model.proj_head.proj.bias is not None) == before
    assert "proj_bias" not in model.get_config()
    model.change_config(proj_norm_before=not before)
    assert (model.proj_head.proj.bias is not None) != before


def test_projection_bias_preserves_legacy_default():
    assert ProjHead(4, norm_before=True).proj.bias is not None
    assert ProjHead(4, norm_before=False).proj.bias is None
    assert ProjHead(4, use_norm=False).proj.bias is not None


@pytest.mark.parametrize("return_feats", [False, True])
def test_forward_masks_padding_and_returns_backbone_features(return_feats):
    model = make_model().eval()
    audio = torch.randn(2, 9)
    lengths = torch.tensor([9, 5])
    target = torch.tensor([0, 1])
    output = model(audio, lengths, target, return_backbone_feats=return_feats)
    assert output.xvector.shape == (2, 3)
    assert output.head_output.logits.shape == (2, 2)
    assert output.head_output.loss.ndim == 0
    short = model(audio[1:2, :5], return_head_output=False)
    torch.testing.assert_close(output.xvector[1:2], short.xvector)
    if return_feats:
        assert output.backbone_output_feats.shape == (2, 9, 4)
        assert len(output.backbone_hidden_feats) == 1
    else:
        assert output.backbone_output_feats is None


def test_config_round_trip_and_input_configs_are_not_mutated():
    pool = {"pool_type": "mean+stddev"}
    head = {"head_type": "classif", "num_classes": 2, "loss_type": "softmax"}
    model = make_model(
        pooling=pool,
        head=head,
        proj_norm_layer="rms-norm",
        proj_use_norm=True,
        proj_norm_before=False,
        pooling_weight_decay=0.1,
        proj_weight_decay=0.2,
        head_weight_decay=0.3,
        bias_weight_decay=0.0,
    ).eval()
    assert pool == {"pool_type": "mean+stddev"}
    assert "in_feats" not in head
    cfg = json.loads(json.dumps(model.get_config()))
    cfg.pop("class_name")
    rebuilt = TinyXVectorP(**cfg).eval()
    rebuilt.load_state_dict(model.state_dict())
    audio = torch.randn(2, 8)
    torch.testing.assert_close(model(audio).xvector, rebuilt(audio).xvector)
    assert model.has_param_groups()
    params = [p for group in model.trainable_param_groups() for p in group["params"]]
    assert len({id(p) for p in params}) == len(list(model.parameters()))


def test_chunk_inference_weights_valid_samples_including_implicit_lengths():
    model = make_model(pooling="avg").eval()
    audio = torch.randn(2, 11)
    full = model(audio, return_head_output=False).xvector
    chunked = model.infer(
        audio, max_batch_duration=0.8, override_chunk_duration=0.4
    ).xvector
    torch.testing.assert_close(chunked, full)
    lengths = torch.tensor([11, 6])
    full = model(audio, lengths, return_head_output=False).xvector
    chunked = model.infer(audio, lengths, override_chunk_duration=0.4).xvector
    torch.testing.assert_close(chunked, full)


def test_output_concatenation_and_weighted_average():
    outputs = [
        XVectorPOutput(xvector=torch.tensor([[1.0, 2.0]])),
        XVectorPOutput(xvector=torch.tensor([[3.0, 4.0], [5.0, 6.0]])),
    ]
    output = XVectorPOutput.weighted_average_by_index(
        XVectorPOutput.concatenate(outputs),
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 3.0, 2.0]),
    )
    torch.testing.assert_close(output.xvector, torch.tensor([[2.5, 3.5], [5.0, 6.0]]))


@pytest.mark.parametrize("mode", XVectorPTrainMode.choices())
def test_training_modes(mode):
    model = make_model(proj_use_norm=True)
    model.set_train_mode(XVectorPTrainMode(mode))
    model.train()
    assert model.train_mode == mode
    assert model.training == (mode != "frozen")
    assert model.pooling.training == (
        mode in ["full", "frozen-feat-extractor", "pooling"]
    )
    assert model.proj_head.training == (mode not in ["frozen", "output-layer"])
    if mode in ["pooling", "proj-head", "output-layer", "frozen"]:
        assert not any(p.requires_grad for p in model.backbone.parameters())
    if mode not in ["frozen", "output-layer"]:
        model(
            torch.randn(2, 8), target=torch.tensor([0, 1])
        ).head_output.loss.backward()
        assert model.proj_head.proj.weight.grad is not None


def test_finetuning_rebuilds_correct_pooling_dimension_and_keeps_eval_state():
    model = make_model(proj_norm_layer="layer-norm", proj_use_norm=True).double().eval()
    model.set_train_mode("output-layer")
    model.change_config(xvector_dim=5, proj_norm_before=False)
    assert model.proj_head.in_feats == 8
    assert model.proj_head.out_feats == model.head.in_feats == 5
    assert model.proj_head.proj.weight.dtype == torch.float64
    assert not model.proj_head.training
    assert not model.proj_head.proj.weight.requires_grad
    assert model(torch.randn(2, 9, dtype=torch.float64)).xvector.shape == (2, 5)


def test_model_without_downstream_head():
    model = make_model(head={"head_type": "none"})
    assert model(torch.randn(2, 8)).head_output is None
    assert model.get_config()["head"] == {"head_type": "none"}
    model.set_train_mode("proj-head")
    model.change_config(xvector_dim=5)
    assert model.head is None


def test_attention_pooling_projection_width():
    model = make_model(
        pooling={
            "pool_type": "scaled-dot-prod-att-v1",
            "num_heads": 2,
            "d_k": 2,
            "d_v": 3,
        }
    ).eval()
    assert model.proj_head.in_feats == 6
    assert model(torch.randn(2, 8)).xvector.shape == (2, 3)


def test_parser_options_match_constructor_and_finetuning():
    parser = ArgumentParser()
    XVectorP.add_class_args(parser, prefix="model")
    cfg = namespace_to_dict(
        parser.parse_args(
            [
                "--model.proj-norm-layer",
                "rms-norm",
                "--model",
                '{"proj_norm_before": false}',
                "--model.proj-weight-decay",
                "0.2",
                "--model.head.num-classes",
                "2",
            ]
        )
    )["model"]
    assert XVectorP.filter_args(**cfg) == cfg
    assert cfg["proj_norm_layer"] == "rms-norm"
    assert not cfg["proj_norm_before"]
    fine = ArgumentParser()
    XVectorP.add_finetune_args(fine, skip={"head", "override_head"})
    cfg = namespace_to_dict(fine.parse_args([]))
    assert cfg["override_sig_reg"] is False
    assert cfg["enable_xvector_sig_reg"] is False
    assert all(
        value is None
        for key, value in cfg.items()
        if key not in {"override_sig_reg", "enable_xvector_sig_reg", "xvector_sig_reg"}
    )
    assert XVectorP.filter_finetune_args(**cfg) == cfg
    skipped = ArgumentParser()
    XVectorP.add_class_args(skipped, skip={"pooling", "head", "proj_norm_layer"})
    assert "proj_norm_layer" not in namespace_to_dict(skipped.parse_args([]))


def test_model_owned_xvector_sig_reg():
    """SIGReg follows model configuration, embedding gradients, and state_dict."""
    model = make_model(enable_xvector_sig_reg=True, xvector_sig_reg={"num_slices": 8})
    output = model(torch.randn(5, 20))
    assert output.xvector_sig_reg is not None
    assert model.xvector_sig_reg.distributed_mode == "global_data"
    output.xvector_sig_reg.backward()
    assert torch.isfinite(model.backbone.weight.grad).all()
    cfg = model.get_config(no_class_name=True)
    assert cfg["enable_xvector_sig_reg"]
    assert cfg["xvector_sig_reg"] == {"num_slices": 8}
    clone = make_model(enable_xvector_sig_reg=True, xvector_sig_reg={"num_slices": 8})
    clone.load_state_dict(model.state_dict())
    assert clone.xvector_sig_reg.counter.item() == 1
    assert make_model()(torch.randn(5, 20)).xvector_sig_reg is None
    parser = ArgumentParser()
    XVectorP.add_class_args(parser)
    cfg = parser.parse_args(
        ["--enable-xvector-sig-reg", "--xvector_sig_reg.num-slices=8"]
    ).as_dict()
    assert cfg["enable_xvector_sig_reg"]
    assert cfg["xvector_sig_reg"]["distributed_mode"] == "global_data"


def test_xvector_sig_reg_forces_global_statistics():
    """Embedding SIGReg always aggregates samples globally."""
    model = make_model(
        enable_xvector_sig_reg=True,
        xvector_sig_reg={"distributed_mode": "local"},
    )
    assert model.xvector_sig_reg.distributed_mode == "global_data"


def test_chunked_inference_computes_sig_reg_once_on_returned_embeddings():
    """Chunk count does not change collective count or sampling progress."""
    from hyperion.torch.losses.sig_reg import SIGReg

    model = make_model(
        enable_xvector_sig_reg=True, xvector_sig_reg={"num_slices": 8}
    ).eval()
    output = model.infer(torch.randn(2, 20), override_chunk_duration=0.4)
    assert model.xvector_sig_reg.counter.item() == 1
    expected = SIGReg(num_slices=8)(output.xvector)
    torch.testing.assert_close(output.xvector_sig_reg, expected)


def test_finetune_sig_reg_override():
    """Overrides reset sampling state and preserve device and global aggregation."""
    model = make_model(
        enable_xvector_sig_reg=True, xvector_sig_reg={"num_slices": 8}
    ).eval()
    model.xvector_sig_reg.counter.fill_(7)
    original = model.xvector_sig_reg
    model.change_config(
        enable_xvector_sig_reg=False, xvector_sig_reg={"num_slices": 16}
    )
    assert model.xvector_sig_reg is original
    assert model.xvector_sig_reg.counter.item() == 7
    model.change_config(
        override_sig_reg=True,
        enable_xvector_sig_reg=True,
        xvector_sig_reg={"num_slices": 16, "distributed_mode": "local"},
    )
    assert model.xvector_sig_reg is not original
    assert model.xvector_sig_reg.num_slices == 16
    assert model.xvector_sig_reg.distributed_mode == "global_data"
    assert model.xvector_sig_reg.counter.item() == 0
    assert model.xvector_sig_reg.t.device == model.proj_head.proj.weight.device
    assert not model.xvector_sig_reg.training
    model.change_config(override_sig_reg=True)
    assert not model.enable_xvector_sig_reg
    assert not hasattr(model, "xvector_sig_reg")
    assert model(torch.randn(2, 8)).xvector_sig_reg is None
    parser = ArgumentParser()
    XVectorP.add_finetune_args(parser)
    cfg = parser.parse_args(
        [
            "--override-sig-reg",
            "--enable-xvector-sig-reg",
            "--xvector_sig_reg.num-slices=8",
        ]
    ).as_dict()
    model.change_config(**XVectorP.filter_finetune_args(**cfg))
    assert model.enable_xvector_sig_reg
    assert model.xvector_sig_reg.num_slices == 8
