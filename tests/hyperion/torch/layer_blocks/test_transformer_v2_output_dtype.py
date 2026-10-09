"""Norm outputs preserve feature dtype when no projection follows them."""

import pytest
import torch
import torch.nn as nn

from hyperion.torch.layer_blocks.transformer_v2 import (
    TransfomerV2Conv1dStemBlock,
    TransfomerV2Conv2dStemBlock,
)
from hyperion.torch.layers.attention_v2 import SDPBackendType
from hyperion.torch.layers.norm_layers import RMSNorm
from hyperion.torch.narchs.qformer_v2 import QFormerV2
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


@pytest.mark.parametrize("architecture", ["encoder", "qformer", "qformer-multi"])
@pytest.mark.parametrize("norm_layer", ["layer-norm", "rms-norm"])
@pytest.mark.parametrize("out_feats", [None, 8])
def test_architecture_final_norm_dtype(
    architecture: str, norm_layer: str, out_feats: int | None
) -> None:
    """Simulate an FP32 norm output under portable CPU autocast.

    Args:
        architecture: Encoder or either QFormer input path.
        norm_layer: Normalization family.
        out_feats: Optional output projection width.
    """
    options = dict(
        in_feats=8,
        out_feats=out_feats,
        num_heads=2,
        num_kv_heads=1,
        ff_multiple_of=4,
        norm_layer=norm_layer,
        pre_post_norm=True,
        sdp_backend=SDPBackendType.MATH,
    )
    if architecture == "encoder":
        model = TransformerEncoderV2(
            **options,
            stem_type="conv1d",
            stem_hidden_channels=[8],
            stem_kernel_sizes=[3],
            stem_strides=[1],
            encb_repeats=[1],
            hidden_dims=[8],
        )
    else:
        model = QFormerV2(
            **options,
            num_layers=1,
            hidden_dim=8,
            cross_att_freq=1,
            multilayer_input=architecture == "qformer-multi",
        )
    model.eval()
    input_dtypes = []

    def force_fp32(
        module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        """Reproduce accelerator LayerNorm promotion without requiring a GPU.

        Args:
            module: Final norm module.
            inputs: Norm input tuple.
            output: Norm output.

        Returns:
            FP32 norm output.
        """
        input_dtypes.append(inputs[0].dtype)
        return output.float()

    handle = model.out_norm.register_forward_hook(force_fp32)
    features = torch.randn(2, 6, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if architecture == "encoder":
            output, _ = model(features)
        else:
            queries = torch.randn(2, 3, 8, dtype=torch.bfloat16)
            output = model(queries, [features] if model.multilayer_input else features)
    handle.remove()
    assert input_dtypes == [torch.bfloat16]
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "stem_class", [TransfomerV2Conv1dStemBlock, TransfomerV2Conv2dStemBlock]
)
@pytest.mark.parametrize("norm_layer", [nn.LayerNorm, RMSNorm])
def test_unprojected_stem_feature_dtype(stem_class: type, norm_layer: type) -> None:
    """The separately returned normalized stem features match projected features.

    Args:
        stem_class: Convolutional stem implementation.
        norm_layer: Normalization constructor.
    """
    stem = stem_class(
        8, 8, hidden_channels=[8], kernel_sizes=[3], strides=[1], norm_layer=norm_layer
    ).eval()
    handle = stem.norm_layer.register_forward_hook(
        lambda module, inputs, output: output.float()
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        normalized, projected, _ = stem(torch.randn(2, 6, 8))
    handle.remove()
    assert normalized.dtype == projected.dtype == torch.bfloat16
