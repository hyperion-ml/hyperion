PyTorch Layers and Architecture Catalog
=======================================

This catalog helps select reusable PyTorch components before writing a new task
model. It is a map of supported package exports, not a requirement to import
every listed class directly. Use the nearest compatible architecture or factory
first; a new task model should compose these pieces and retain the serialization
contract in :doc:`torch-extension-points`.

Input representation
--------------------

Choose components according to the representation at the point where they are
used:

* waveforms are generally shaped ``(batch, num_samples)``;
* frame sequences generally have a time and feature/channel axis, with the
  exact layout defined by their model or frontend; and
* 2-D acoustic inputs carry channel, frequency, and time axes.

Pass valid lengths or masks through the model whenever the implementation
supports them. Do not infer padding from zero-valued samples or frames: zero
can be valid audio or an ordinary feature value.

Frontends, normalization, and pooling
-------------------------------------

.. autoclass:: hyperion.torch.layers.audio_feats_factory.AudioFeatsFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.layers.norm_layer_factory.NormLayer1dFactory
   :no-index:
   :members: create

.. autoclass:: hyperion.torch.layers.pool_factory.GlobalPool1dFactory
   :no-index:
   :members: create, filter_args, add_class_args

``AudioFeatsFactory`` creates waveform-to-feature operations such as
log-filterbanks and MFCCs. Normalization and pooling factories are the
configuration-facing choices for architectures; they avoid hard-coding a
normalization or utterance-pooling policy into a task model. The exported
pooling layers include mean, mean/std, attention-weighted, log-variance, and
LDE-style variants.

For pretrained transformer frontends, use the wrappers described in
:doc:`torch-integrations-and-robustness`, rather than combining a raw external
model with internal pooling code.

Reusable blocks
---------------

``hyperion.torch.layer_blocks`` provides compositions used by architecture
classes. The main families are:

* TDNN and extended TDNN blocks for frame sequences;
* ResNet/Res2Net, squeeze-excitation, and ConvNeXt blocks for 1-D and 2-D
  encoders or decoders;
* Conformer and Transformer encoder blocks for context-aware sequences;
* FC, MBConv, SpineNet, and DC encoder/decoder blocks for specialised
  architectures; and
* projection, classifier, and Hydra heads when a model needs one or more task
  outputs.

Layer blocks are implementation-level ``torch.nn.Module`` components. Reuse
their accepted tensor layout and constructor options from the architecture that
already uses them. A block is not usually a serializable deployment artifact by
itself.

Neural architecture families
----------------------------

The following architecture families are stable building blocks for conventional
speech and speaker-recognition models:

``TDNNV1`` and ``ETDNNV1``
  Frame-sequence encoders. Their respective factories expose maintained
  configuration selections.

``ResNet``, ``ResNet1dEncoder``, and ``ResNet2dEncoder``
  Residual encoders for one- or two-dimensional acoustic representations.
  The ResNet factory selects the supported named variants.

``ConformerEncoderV1`` and ``TransformerEncoderV1``/``TransformerEncoderV2``
  Context-aware sequence encoders. Specify attention context and mask/length
  behavior consistently with the enclosing model.

``ConvNext1dEncoder``, ``ConvNext2dEncoder``, ``EfficientNet``, and ``SpineNet``
  Convolutional alternatives with their associated factory/configuration
  interfaces.

``ClassifHead``, ``ProjHead``, ``HydraHead``, and ``FeatFuserMVN``
  Output, multi-task, projection, and feature-fusion components. Select them
  in the task model, where loss and target semantics are known.

All architecture classes derive from the shape-reporting contract described in
:doc:`torch-extension-points`. A model that combines an encoder with a pooling
and classification head should document the intermediate layout at each
boundary, especially when it exposes embedding extraction.

Local/global attention in TransformerEncoderV2
----------------------------------------------

``TransformerEncoderV2`` generates a local/global schedule using
``local_to_global_ratio`` (default ``0``, all global). A ratio of ``5`` repeats
five local layers followed by one global layer. The schedule continues across
superblocks, and the final transformer layer is always global.

``local_attention_sliding_window`` and ``global_attention_sliding_window`` both
default to ``None`` (unrestricted attention). Finite windows require
``att_type="hf_flash_sdp"``; when both are finite and local layers are enabled,
the global window must exceed the local window. Window sizes are measured in
stage tokens. The same window covers more audio after downsampling.

``local_head_dim=None`` and ``global_head_dim=None`` derive each attention
head width from the stage hidden dimension divided by ``num_heads``. Explicit
positive widths allow the internal attention dimension to differ from the
residual stream: Q projects to ``num_heads * head_dim``, K/V project to
``num_kv_heads * head_dim``, and the output projection returns to the stage's
hidden dimension. RoPE requires even head widths. Both options are exposed in
the CLI and saved in the encoder configuration.

Each stage has separate local and global ``RotaryPosEncoder`` instances, so
caches do not mix head dimensions or temporal resolutions. Defaults are:

* ``local_rope_theta=10000.0`` and ``global_rope_theta=1000000.0``;
* ``local_rope_partial_rotary_factor=1.0`` and
  ``global_rope_partial_rotary_factor=1.0``; and
* ``local_rope_scale_freqs=True`` and ``global_rope_scale_freqs=True``.

Partial rotation retains full-head frequency spacing. For head dimension
``d``, pair ``i`` has frequency ``theta ** (-2*i/d)``; only the first
``floor(partial_rotary_factor*d/2)`` pairs rotate. At least one pair must
rotate. Unrotated pairs pass through unchanged. To select Gemma-style
positional encoding, use global fraction ``0.25`` and disable both frequency
scaling flags. The global theta alone does not reproduce all Gemma settings.

The remaining RoPE arguments are shared: ``rope_update_max_seq_length=True``,
``rope_original_max_seq_length=None`` (initial reference length ``8192``),
``rope_scaling_factor=8``, ``rope_low_freq_factor=1``, and
``rope_high_freq_factor=4``. Scaling is applied in training and evaluation when
enabled, regardless of current sequence length. Training can grow each
stage/type's reference length; that rebuilds its rotation cache. For fixed
frequency behavior, including incremental attention with cached keys, disable
reference-length updates. Previously, training with updates enabled bypassed
frequency scaling; this behavior is corrected in the shared positional layer,
including QFormerV2.

These encoder options replace ``att_sliding_window``, ``rope_theta``, and
``rope_scale_freqs`` in its config and CLI. QFormerV2 retains its existing
architecture options. All new options are serialized and exposed by the
encoder argument parser.

Pre/post normalization in V2 Transformers
----------------------------------------

``TransformerEncoderV2`` and ``QFormerV2`` accept ``pre_post_norm=False``
(default), also exposed as ``--pre-post-norm``. This flag controls normalization
placement; ``norm_layer`` independently selects LayerNorm or RMSNorm.
The flag and normalization type are saved in the architecture configuration.

With ``pre_post_norm=False``, attention and feed-forward branches use pre-norm.
With ``True``, the projected self-attention output is normalized before residual
addition. QFormer cross-attention outputs are normalized in the same position.
Dense MLP and ConvNeXt feed-forward outputs are normalized after the final
projection and before residual addition. The ConvNeXt internal normalization is retained and uses the configured
``norm_eps``.

For example, a dense branch follows these equations when the flag is enabled::

    h = x + att_post_norm(attention(att_norm(x)))
    y = h + ff_post_norm(feed_forward_without_post_norm(ff_norm(h)))

Each pre/post norm has independent learned parameters and uses the configured
``norm_layer``. The normalization placement matches Gemma 4; choosing RMSNorm
also matches its normalization type. ``pre_post_norm=True`` does not override
``norm_layer``, enable QK normalization, or change the activation.

QK normalization in V2 Transformers
----------------------------------

``TransformerEncoderV2`` and ``QFormerV2`` accept ``enable_qk_norm=True``
(default: ``False``), also exposed as ``--enable-qk-norm`` in their argument
parsers. The option is saved in the architecture configuration.

When enabled, separate learned RMSNorm layers normalize projected queries and
keys over each head's feature dimension before rotary positional encoding.
They use the architecture's ``norm_eps``, independently of its ``norm_layer``
selection. Attention scores use a multiplier of one instead of
``1 / sqrt(head_dim)`` across all V2 attention backends. Values are unchanged unless ``enable_v_norm=True``.
In ``QFormerV2`` this applies to both self-attention and cross-attention,
including tied layers.

QFormerV2 attention head dimensions
-----------------------------------

``QFormerV2(head_dim=None)`` derives its attention head width from
``hidden_dim / num_heads``. An explicit positive width applies to every
self-attention and cross-attention branch, including tied layers. Internal
attention width can differ from the residual width, while output projections
return to ``hidden_dim``. RoPE requires an even head width. The option is saved
in the architecture configuration and exposed as ``--head-dim``.

Key/value projection reuse in V2 Transformers
--------------------------------------------

``TransformerEncoderV2(global_k_eq_v=False)`` optionally reuses raw key
projections for values in global layers only. Local layers retain separate
value projections. QFormerV2 independently controls its branches with
``self_att_k_eq_v=False`` and ``cross_att_k_eq_v=False``, including tied layers.
CLI options are ``--global-k-eq-v``, ``--self-att-k-eq-v``, and
``--cross-att-k-eq-v``. All options are serialized.

When enabled, attention omits ``v_proj`` and uses the raw ``k_proj`` output
for V before K normalization and RoPE. ``enable_v_norm`` independently controls
value RMSNorm without learned scaling. The final K and V tensors can therefore
differ and retain separate inference caches. This option shares projection
work within a layer; it does not share K/V across layers. To match Gemma's
normalization alongside projection reuse, enable QK and value normalization.

Value normalization in V2 Transformers
-------------------------------------

``TransformerEncoderV2`` and ``QFormerV2`` accept ``enable_v_norm=False`` (default),
also exposed as ``--enable-v-norm`` and saved in the architecture configuration.
When enabled, values receive per-head RMSNorm after projection and before
attention or cache writes. This normalization has no learned scale and uses
``norm_eps``. It is independent of ``enable_qk_norm`` and ``norm_layer``;
values never receive RoPE, and enabling value normalization does not change
attention score scaling. The common attention path applies it to manual,
Torch SDPA, and HF Flash attention, including QFormer cross-attention and tied
layers.

``RMSNorm(dim, eps=1e-6, with_scale=True)`` retains its existing learned scale
by default. ``with_scale=False`` computes RMS normalization without parameters,
using FP32 statistics and returning the input dtype. It does not register a
weight tensor in the state dictionary.

Gemma 4 mixture of experts
--------------------------

``TransformerV2G4MoEBlock`` combines an always-active dense gated MLP with
sparse routed gated-MLP experts. Select it in ``TransformerEncoderV2`` or
``QFormerV2`` with ``ff_type="g4moe"`` and provide:

* ``ff_num_experts``: total number of routed experts;
* ``ff_top_k_experts``: experts selected per token; and
* ``ff_moe_intermediate_dim``: intermediate width of each expert.

The existing ``ff_dim_multiplier`` controls the dense MLP width. Both widths
are rounded using ``ff_multiple_of``. Set ``ff_act="gelu-tanh"`` to match Gemma
4; the architecture-level default activation remains ``"silu"``. Standalone
``TransformerV2G4MoEBlock`` instances default to ``"gelu-tanh"``.

For example, the following options configure either architecture::

    ff_type="g4moe"
    ff_num_experts=8
    ff_top_k_experts=2
    ff_moe_intermediate_dim=512
    ff_act="gelu-tanh"

The corresponding parser options are ``--ff-type``, ``--ff-num-experts``,
``--ff-top-k-experts``, ``--ff-moe-intermediate-dim``, and ``--ff-act``.
These options are included in the saved architecture configuration.

The router applies unscaled RMS normalization, a learned feature scale, and
``1 / sqrt(hidden_dim)`` before a bias-free projection. It selects experts
from float32 softmax probabilities, renormalizes the selected probabilities,
and applies learned per-expert scales. Only selected tokens are evaluated by
each expert; tokens are not dropped.

The dense and expert branches have separate pre-norms. The MoE class defaults
to ``pre_post_norm=False``; setting it to ``True`` adds separate branch
post-norms and a final norm on their sum. Architecture wrappers forward their
``pre_post_norm`` and ``norm_layer`` settings to the MoE, so all branch norms
use the selected normalization type and ``norm_eps``. Standalone MoE instances
default to RMSNorm. Router normalization remains unscaled RMS normalization,
independently of the branch normalization type. Transformer wrappers bypass their
usual feed-forward pre-norm so that the MoE receives the residual stream
before normalization. The standalone feed-forward module returns the branch
output; the wrapper adds the residual.

``model_parallel=True`` shards the dense/expert MLP intermediate projections
using the existing tensor-parallel layers. The router is replicated; experts
are not distributed to separate devices by expert parallelism.

Sparse routing can leave some expert parameters unused in a training step.
DDP training therefore requires unused-parameter detection, for example
``DistributedDataParallel(..., find_unused_parameters=True)``. The standard
``TorchTrainerBase`` DDP path currently does not enable this option; using
``g4moe`` there requires additional trainer integration.

Factories and selection
-----------------------

.. autoclass:: hyperion.torch.narchs.resnet_factory.ResNetFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.narchs.tdnn_factory.TDNNFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.narchs.spinenet_factory.SpineNetFactory
   :no-index:
   :members: create, filter_args, add_class_args

Use a factory at a CLI/configuration boundary, then save the resolved model
configuration with the experiment. Factories centralize aliases and valid
arguments; copying their selection logic into a command makes configuration
compatibility harder to preserve.

Experimental architecture families
----------------------------------

DAC/streaming-DAC layers and architectures, transducer decoder/predictor
blocks, and vector-quantization layers used by q-vector or codec workflows are
experimental in those contexts. They are described in
:doc:`experimental-components`; do not assume their architecture names,
checkpoint compatibility, or output-code semantics are stable.

See also
--------

* :doc:`torch-api`
* :doc:`torch-extension-points`
* :doc:`experimental-components`
