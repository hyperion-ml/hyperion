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
stage/type's reference length; that rebuilds its rotation cache. Evaluation never grows the reference, so inference caches retain a fixed
rotation after training. Disable reference-length updates for fixed training
frequencies as well; caching while training with a growing reference is unsupported. Previously, training with updates enabled bypassed
frequency scaling; this behavior is corrected in the shared positional layer,
including QFormerV2.

These encoder options replace ``att_sliding_window``, ``rope_theta``, and
``rope_scale_freqs`` in its config and CLI. QFormerV2 retains its existing
architecture options. All new options are serialized and exposed by the
encoder argument parser.

Pre/post normalization in V2 Transformers
-----------------------------------------

``TransformerEncoderV2`` and ``QFormerV2`` accept ``pre_post_norm=False``
(default), also exposed as ``--pre-post-norm``. This flag controls normalization
placement; ``norm_layer`` independently selects LayerNorm or RMSNorm.
The flag and normalization type are saved in the architecture configuration.
``norm_eps=1e-5`` is passed to branch, stem, endpoint, output, Q/K/V, and MoE
normalizations; Q/K/V and router norms remain RMS-based regardless of
``norm_layer``.

With ``pre_post_norm=False``, attention and feed-forward branches use pre-norm.
With ``True``, the projected self-attention output is normalized before residual
addition. QFormer cross-attention outputs are normalized in the same position.
Dense MLP and ConvNeXt feed-forward outputs are normalized after the final
projection and before residual addition. The ConvNeXt internal normalization
is retained and uses the configured ``norm_eps``.

For example, a dense branch follows these equations when the flag is enabled::

    h = x + att_post_norm(attention(att_norm(x)))
    y = h + ff_post_norm(feed_forward_without_post_norm(ff_norm(h)))

Each pre/post norm has independent learned parameters and uses the configured
``norm_layer``. The dense attention/MLP branches use Gemma-style normalization placement;
choosing RMSNorm also selects its normalization type. ConvNeXt retains its
own internal convolution and normalization structure. ``pre_post_norm=True`` does not override
``norm_layer``, enable QK normalization, or change the activation.

QK normalization in V2 Transformers
-----------------------------------

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
---------------------------------------------

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

Shared K/V inputs in V2 attention
---------------------------------

``ScaledDotProdAttV2``, ``TorchScaledDotProdAttV2``, and
``HFFlashScaledDotProdAttV2`` accept ``shared_kv=False``. With ``shared_kv=True``,
the existing ``key`` and ``value`` forward inputs must contain processed source
tensors shaped ``(batch, key_time, local_kv_heads, head_dim)``. Keys must already
include source normalization and RoPE; values must already include any source
value normalization. The consumer projects, normalizes, and rotates only Q,
then runs its usual attention backend and output projection. It has no K/V
projections or normalization modules, even when ``k_eq_v`` or value normalization
is enabled. Those operations belong to the source layer.

Shared layers cannot allocate or update their own cache. Pass the valid source
cache slices as K/V, leave ``state=None``, and provide any mask needed for cache
offsets or causal constraints. ``query_start_pos`` controls query RoPE;
``key_start_pos`` is ignored for shared inputs. Gradients flow through source
K/V during training. ``return_kv=True`` exports
``(output, processed_key, processed_value, updated_state)`` from attention;
without caching, ``updated_state`` is None. The self-attention block similarly
exports ``(output, (processed_key, processed_value), updated_state)``.

``TransformerEncoderV2(num_kv_shared_layers=0)`` configures shared suffixes
within its superblocks. An integer applies the same suffix count to every
superblock; a list provides one count per superblock. Each consumer reuses the
last non-shared layer of its own local/global attention type in that superblock.
Sources remain fixed throughout the shared suffix. Counts must leave at least
one independent layer and an independent source for each consumed attention
type. Sharing never crosses superblock boundaries, including dimension changes
or downsampling. The local/global attention schedule itself remains unchanged.

For example, six layers with a 1:1 local/global schedule and
``num_kv_shared_layers=2`` share the final local/global layers with layers 2/3
(zero-based). Consumers keep their own Q and output projections, attention
pre/post norms, and feed-forward branches. ``global_k_eq_v`` and K/V norms are
applied by the global source. The CLI accepts ``--num-kv-shared-layers=2`` or
``--num-kv-shared-layers='[0, 2]'``; configuration serializes per-superblock counts.
Cache state retains one entry per layer, with ``self_att=None`` for consumers;
only independent layers allocate and update K/V buffers. Forward sharing uses
local tensor references, so gradients reach the source during training and no
source tensors are stored on the model between forward calls. QFormerV2 does
not configure shared suffixes.

Encoder attention masks and streaming
-------------------------------------

``TransformerEncoderV2(is_causal=True)`` combines key padding with causality
for manual and Torch attention. Explicit masks use boolean keep semantics and
shape ``(batch, 1, query_time, key_time)``; causal comparisons use absolute
query/key positions, including the visible cache offset. Torch's native causal
shortcut is retained for uncached calls without a padding mask. HF Flash uses
its usual two-dimensional padding mask and separate causal flag.

For cached calls, historical keys are treated as valid for active sequences,
and padding is applied to the current chunk. A partially valid chunk must be
final: subsequent calls use zero length for that batch element, its outputs
are ignored, and it must not resume. This contract requires no extra cache
fields or persistent historical masks. Masks account for cache rollover and
stage downsampling. Each superblock builds masks for local and global histories
before source caches update, and reuses a mask when the histories match.
Shared consumers use the mask built for their source's attention type.
Independent caches of the same type within a superblock must have matching
lengths, offsets, and capacities.

Cache capacity is ``min(stage_cache_length, sliding_window)`` for layers with
a finite attention window, including global layers when a global window is
configured. Layers without a window use ``stage_cache_length``. The stage
limit is derived from ``init_state(max_cache_length=...)`` and inter-stage
downsampling. Windows stay in each stage's token units; they are not rescaled.
Shared layers allocate no separate cache. Attention K/V tensors start empty;
``CacheState["max_cache_length"]`` records the retention limit separately from
their current size. Each update concatenates retained history with current
K/V, then stores suffix views without copying into preallocated buffers.
A retained view keeps the full attention allocation alive until replaced by
the next update and all other references have been released.
Cached attention uses the previous history and the entire current chunk, even
when the chunk exceeds cache capacity. After attention, only the last
capacity-sized suffix is retained for the next call. Shared layers consume
the full K/V used by their source layer, including the entire current chunk.
Causal encoders use ``TransformerEncoderV2StreamingConv1dStemBlock`` and
``TransformerV2StreamingConvDownsampleBlock`` in place of ordinary temporal
convolutions. Downsampling blocks accept optional ``x_lengths`` in both paths:
``forward`` returns ``(x, x_lengths)``, while ``stream`` additionally returns
the updated convolution state. Full-sequence calls without state use causal ``forward`` methods
with gradients enabled. Calls with state use convolution ``stream`` methods
for inference, carrying input history and stride phase alongside attention
caches in ``TransformerEncoderState.stem_state`` and ``downsample_states``.
Use ``eval()`` and ``torch.no_grad()`` for streaming inference, and advance
``start_pos`` by the number of input frames consumed, including chunks that
emit no output. Stride phase determines exact output lengths and positions;
chunks need not be multiples of the downsampling factor. A stage emitting
no frames leaves its attention caches unchanged. Causal convolution right
context is zero.

Conv2D stems and ConvNeXt feed-forward blocks are rejected when
``is_causal=True``. Multilayer endpoints must already have the target temporal
scale; endpoint resampling is not supported for causal streaming. Ordinary
stems, downsampling and endpoints remain available for non-causal encoders.

Value normalization in V2 Transformers
--------------------------------------

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
using FP32 statistics for FP16/BF16 inputs. Scaling is applied before the final
cast, so FP32 weights do not promote low-precision outputs outside autocast.
Like native LayerNorm, it preserves the input dtype on CPU and outside
autocast, but returns FP32 for non-FP64 inputs under CUDA/MPS/XPU autocast.
FP64 inputs use FP64 computation. Without scaling, it does not register a
weight tensor in the state dictionary.

V2 attention and branch post-norms cast normalized outputs back to their
incoming dtype before attention or residual addition. Encoder and QFormer
final norms do the same when no output projection follows; otherwise the
projection determines the compute dtype under autocast. Both convolutional
stems return unprojected normalized features in the projected feature dtype.
Pre-norms followed by linear or convolutional projections rely on those
projections' autocast behavior.

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
Both architectures report this through
``requires_ddp_find_unused_parameters()`` when ``ff_type="g4moe"``.
``TorchTrainerBase`` and the legacy trainer pass the top-level model's result
to ``DistributedDataParallel(find_unused_parameters=...)``. A composed task
model must override and delegate this method to its encoder/QFormer; the
trainer does not scan child modules. See :doc:`torch-training-support`.


Attention backends and Flash Attention versions
-----------------------------------------------

Both architectures accept ``att_type="sdp"`` (manual), ``"torch_sdp"`` (default),
or ``"hf_flash_sdp"``. ``sdp_backend`` selects native Torch SDPA backends and
is saved in the configuration; the backend lists retain a math fallback.
``flash_attention_version=None`` selects the backend default: native Torch
FA2 or HF FA4. Explicit values ``2``, ``3``, or ``4`` propagate to every HF
attention branch, including QFormer self-attention-only blocks and both
branches of cross-attention blocks. The option is saved and exposed as
``--flash-attention-version``. Standalone
``HFFlashScaledDotProdAttV2`` defaults to FA4.

Native Torch version selection changes process-wide state through
``TorchScaledDotProdAttV2.set_flash_attention_version``. PyTorch at or below
2.9.1 accepts only FA2. Newer builds must provide the implementation registry
and the requested kernels; unavailable versions raise ``RuntimeError``.
Constructing another architecture can change the selection for existing
native SDPA modules in the same process. The repository dependency range
currently caps Torch at 2.9.1, so native FA3/FA4 require a separately validated
newer runtime. Selecting a version does not force every SDPA call to use
Flash Attention: dtype, device, masks, and the selected backend list still
control dispatch.

HF Flash Attention is conditional on a compatible Transformers version,
installed kernels, supported device, and compute dtype. The implementation
uses HF's private ``_flash_attention_forward`` helper; FA2/FA3/FA4 dispatch and
variable-length cross-attention still need runtime testing with newer HF
versions before changing the dependency range. Do not interpret the HF FA4
default as a compatibility guarantee for every supported Transformers release.

HF self-attention accepts boolean key-padding masks shaped ``(B, K)``.
Numeric masks use additive values: nonnegative entries are valid, negative
entries are masked. Non-causal cross-attention also accepts padding-only
``(B, 1, Q, K)`` masks whose rows are identical. Pairwise masks are unsupported.
For padded cross-attention with different Q/K lengths, the wrapper packs all
valid queries and only valid keys with separate cumulative sequence lengths;
queries are not truncated to the key lengths.

QFormer positional encoding and valid inputs
--------------------------------------------

``rope_in_self_att=False`` and ``rope_in_cross_att=False`` are independent
QFormer defaults. Leaving cross-attention RoPE disabled is appropriate for
learned query slots and acoustic frames without a shared positional timeline.
Self-attention-only blocks honor ``rope_in_self_att`` even when cross-attention
RoPE is enabled. QFormer has a single ``rope_theta`` and scaling configuration;
it does not create local/global layer schedules or shared KV suffixes.

Every QFormer example should contain at least one valid encoder key. Fully
masked manual attention is outside this usage contract. If cross-attention
RoPE is enabled during training, a K sequence that grows the dynamic reference
after Q has rotated can give Q and K different frequencies. Use a fixed
reference (``rope_update_max_seq_length=False``) or disabled frequency scaling
for that configuration; the default disabled cross-attention RoPE avoids it.

Exact lengths, padding, and multi-layer endpoints
-------------------------------------------------

Non-causal convolution lengths use the actual kernel, stride, padding, and
dilation, preserving zero valid lengths. ``conv_output_lengths`` in
``hyperion.torch.utils.misc`` exposes the same calculation for reuse. Causal
streaming instead uses convolution stride phase. Ordinary downsampling
``forward`` returns ``(features, lengths)`` even when lengths are omitted.

Stems and downsampling blocks clear padding immediately before temporal
convolutions, including after normalization. Feed-forward blocks accept an
optional ``x_mask`` with boolean keep or additive semantics, shaped ``(B, T)``
or ``(B, 1, Q, K)``. Pointwise dense/MoE blocks ignore it; ConvNeXt clears
padding before its depthwise convolution and excludes it from GRN. ConvNeXt
requires a positive odd kernel size to preserve residual length and rejects
``model_parallel=True``.

With ``multilayer=True``, ``endpoint_layers=None`` selects all superblocks.
An explicit nonempty list selects zero-based superblock indices; the final
superblock is always appended if absent, keeping every stage connected to the
aggregated training output. The effective list is saved in ``get_config()``.
``endpoint_scale_layer=-1`` chooses the final stage's temporal scale by default.

Endpoint blocks receive right-padding masks in either 2-D or 4-D form. They
clear padding after normalization and intermediate projections, and use
negative infinity to exclude invalid positions from max-pooling, including
when valid features are negative. Masks and lengths follow each convolution,
pooling, and interpolation operation. Endpoint tensors are center-cropped to
the shortest resampled length before averaging or concatenation. Output valid
lengths subtract each left crop, clamp to the merged size, and take the minimum
across endpoints. ``out_shape`` accounts for the same resampling and cropping,
rather than always reporting the final stage size. See
:doc:`torch-api-contracts` for forward return values and cache mutation.

Tensor-parallel attention requires positive Q/KV head counts, Q heads divisible
by KV heads, and both counts divisible by the tensor-parallel world size.
Incompatible configurations raise ``ValueError`` before distributed projections
are created; KV replication across ranks is not implemented.

GELU activation choice
-----------------------

``hyperion.torch.layers.activation_factory.ActivationFactory`` exposes
``"gelu-tanh"`` as ``torch.nn.GELU(approximate="tanh")``. ``"gelu"`` retains the
non-approximate GELU implementation. Both are gated-MLP activation choices;
the activation name is saved as ``ff_act`` in architecture configurations.

Configure an encoder and QFormer on synthetic frames
-----------------------------------------------------

This CPU example needs an installed Hyperion runtime with Torch and
Transformers. It uses no downloaded model or corpus. The encoder emits frame
features and valid lengths; QFormer converts them to four query features per
example. Shapes depend on the exact convolution lengths.

.. code-block:: python

   import torch
   from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2
   from hyperion.torch.narchs.qformer_v2 import QFormerV2

   encoder = TransformerEncoderV2(
       in_feats=8, stem_type="conv1d", stem_hidden_channels=[16],
       stem_kernel_sizes=[3], stem_strides=[1], stem_dropout_rate=0.0,
       encb_repeats=[4], hidden_dims=[16], downb_strides=[], num_heads=4,
       ff_multiple_of=8, ff_type="g4moe", ff_num_experts=4,
       ff_top_k_experts=2, ff_moe_intermediate_dim=32, ff_act="gelu-tanh",
       enable_qk_norm=True, enable_v_norm=True, pre_post_norm=True,
       norm_layer="rms-norm", local_to_global_ratio=1,
       num_kv_shared_layers=2, global_k_eq_v=True,
       local_rope_scale_freqs=False, global_rope_scale_freqs=False,
   ).eval()
   qformer = QFormerV2(
       in_feats=16, hidden_dim=16, num_heads=4, num_layers=2,
       ff_multiple_of=8, enable_qk_norm=True, enable_v_norm=True,
       pre_post_norm=True, norm_layer="rms-norm", head_dim=8,
       rope_in_self_att=False, rope_in_cross_att=False,
   ).eval()
   frames = torch.randn(2, 20, 8)
   queries = torch.randn(2, 4, 16)
   with torch.no_grad():
       features, lengths = encoder(frames, torch.tensor([20, 16]))
       query_features = qformer(queries, features, lengths)
   assert features.shape == encoder.out_shape(tuple(frames.shape))
   assert query_features.shape == (2, 4, 16)

For causal inference, construct the encoder with ``is_causal=True``, call
``state = encoder.init_state(batch_size, max_cache_length)``, and pass state
through ``output, lengths, state = encoder(chunk, chunk_lengths,
start_pos=consumed_frames, state=state)``. ``max_cache_length`` is measured at
the first transformer stage after the stem. Persist configuration and weights
through the model's standard ``save``/``load`` interface; ``get_config`` records
architecture settings but does not replace ``state_dict``. Inference cache
state is external and is not included in the architecture checkpoint.

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
