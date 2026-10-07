SIGReg Gaussian regularization
==============================

``hyperion.torch.losses.SIGReg`` implements the Epps--Pulley version of
LeJEPA Algorithm 1: random unit projections, empirical characteristic-function
matching, and Gaussian-windowed trapezoidal integration. Defaults are 256
directions and 17 equally spaced frequencies over [-5, 5]. It preserves the
global sample-count multiplier and averages over directions. Raw embeddings
are used without centering, whitening, standardization, or normalization.

For raw samples :math:`z_n \in \mathbb{R}^D`, sample unit directions
:math:`u_m` and set :math:`h_{n,m}=z_n^\top u_m`. With global sample count
:math:`N`, the implemented objective is

.. math::

   \widehat\phi_m(t) = \frac{1}{N}\sum_{n=1}^{N}e^{it h_{n,m}},
   \qquad \phi_0(t)=w(t)=e^{-t^2/2},

.. math::

   L_{\mathrm{SIGReg}} = \frac{N}{M}\sum_{m=1}^{M}
   \operatorname{trapz}_t\!\left[
       w(t)\left|\widehat\phi_m(t)-\phi_0(t)\right|^2
   \right].

The cached integration weights combine the trapezoidal coefficients (half
weight at each endpoint) with the Gaussian window. This is equivalent to
``torch.trapz(error * exp(-t**2 / 2), t)``. Finite Gaussian batches do not
have exactly zero loss. The method follows
`LeJEPA Algorithm 1 <https://arxiv.org/pdf/2511.08544>`_.

.. code-block:: python

   from hyperion.torch.losses import SIGReg

   embedding_reg = SIGReg(num_slices=1024, distributed_mode="global_data")
   prototype_reg = SIGReg(distributed_mode="local")
   embedding_loss = embedding_reg(embeddings)  # [B, D]
   prototype_loss = prototype_reg(prototypes)  # raw weights [classes, D]

Input shapes and sequences
--------------------------

With ``multi_view=False``, inputs are [B, D] or [B, T, D]. With
``multi_view=True``, inputs are [V, B, D] or [V, B, T, D]. Each view is
regularized separately with shared directions; view losses are averaged by default.
Views are never pooled together or concatenated along the embedding axis.

``reduction="mean"`` (default) averages view losses, ``"sum"`` sums them,
and ``"none"`` returns a vector [V]. Single-view inputs always return a scalar.
Projection directions are always averaged and each view retains its own
global valid-sample multiplier, regardless of the reduction setting.

Sequence inputs accept ``torch.long`` ``z_lengths`` of shape [B] or [V, B],
already on the same device as ``z``. Each sequence contributes its first
length frames; omitted lengths include every time step. Valid frames are pooled
within a view. Longer sequences contribute
more samples, and the global valid-frame count multiplies each view's loss.
Neighboring frames may be correlated: this objective matches the pooled frame
distribution rather than imposing Gaussian statistics within each sequence.

Distributed gradients and sampling
----------------------------------

``global_data`` reduces local sine/cosine sums with an autograd-aware SUM
collective, and separately sums sample counts. This supports unequal local
counts and empty local views when the global view is nonempty. Communication
contains only projection/frequency statistics and counts. Without initialized
distributed training, it behaves as a single-process loss.

Every rank evaluates the same global objective. The collective's backward SUM
produces a world-size factor in local embedding gradients; standard DDP's
parameter-gradient averaging cancels it. Do not additionally scale the loss
by world size. ``local`` performs no communication and is suitable for identical
replicated trainable prototypes evaluated on every rank.

A dedicated generator is created lazily on the input device and reused. Its
seed is the configured base seed plus a counter. Directions resample once per
forward call, including gradient-accumulation microbatches and evaluation calls.
The counter is saved in ``state_dict``. All ranks must have matching seeds,
counters, configurations, device types, and call order. Restore the loss state
alongside model state when resuming. Application RNG state is unaffected.

Numerical precision and memory
------------------------------

Projection, trigonometry, and integration run in float32 with autocast disabled,
while gradients propagate to the original embeddings. ``projection_chunk_size``
optionally limits the directions computed together and checkpoints local
statistics to reduce activation memory during training. Collectives remain
outside checkpointed computations. Runtime and memory are linear in sample
count for fixed embedding dimension, direction count, and frequency count.

.. autoclass:: hyperion.torch.losses.SIGReg
   :members: forward, apply_reduction, get_config, filter_args, add_class_args
   :exclude-members: training

Hydra classification prototypes
-------------------------------

``HydraClassifHead`` optionally computes SIGReg for its raw class prototypes
when ``enable_prototype_sig_reg=True``. Pass regularizer constructor settings
in the ``prototype_sig_reg`` dictionary, for example
``{"num_slices": 1024, "projection_chunk_size": 256}``.
The statistics mode is fixed to ``local`` because DDP ranks hold replicated
prototype parameters. Cosine heads use their unnormalized kernels; subcenter
heads use only the main (most-used) center of each class. Subcenter usage-count
increments are summed across distributed ranks during labeled training forwards,
so main-center selection agrees across ranks. All ranks must participate in these
updates. The result is returned as
``HydraClassifHeadOutput.prototype_sig_reg`` separately from cross-entropy,
so callers can weight it in their training objective. The enable flag and
nested dictionary are included in the head configuration and CLI parsers.

``XVectorPTrainer`` and ``QVectorTrainer`` add the prototype SIGReg statistic
to the training objective with ``prototype_sig_reg_weight`` (default ``0.0``).
Enable the head regularizer and set a positive trainer weight to train with it.
Both trainers log the available statistic as ``prototype_sig_reg``.

``XVectorP`` calculates SIGReg on its projected xvectors when
``enable_xvector_sig_reg=True``. Configure the module through the model's
``xvector_sig_reg`` dictionary; statistics are fixed to ``global_data`` across
DDP ranks. The result is returned in ``XVectorPOutput.xvector_sig_reg``, and
sampling progress is included in the model state dictionary. The trainer only
weights this value with ``xvector_sig_reg_weight`` (default ``0.0``) and logs it
as ``xvector_sig_reg``. Calculation is controlled by the model flag independently
of the trainer weight.

For fine-tuning, ``XVectorP.change_config(override_sig_reg=True, ...)`` replaces
the embedding regularizer using ``enable_xvector_sig_reg`` and ``xvector_sig_reg``.
A replacement starts with a fresh sampling counter on the model device and
always uses ``global_data``. Setting the enable flag to false disables it.
With ``override_sig_reg=False`` (the default), the existing module, configuration,
and sampling progress are preserved. These options are exposed by the
fine-tuning parsers of XVectorP and its child models.

Code-rate configuration and prototype geometry
----------------------------------------------

Hydra prototype code rate uses ``enable_prototype_code_rate`` and a nested
``prototype_code_rate`` dictionary instead of the removed ``code_rate_eps``
argument. The loss receives ``raw_prototypes`` and forces ``normalize=True``
and ``distributed_mode="local"``. Normalization happens inside the code-rate
loss; SIGReg receives the same raw centers and performs no length normalization.

In ``CosLossOutput``, ``ArcLossOutput``, and ``SubCenterArcLossOutput``,
``raw_prototypes`` and ``prototypes`` both return [num_classes, D]. The first
returns raw centers; the second returns length-normalized centers. Stored kernels
are [D, num_classes], so both accessors transpose the class-center matrix.
Subcenter accessors select main centers first. Hydra exposes ``raw_prototypes``
and ``normalized_prototypes`` uniformly; its existing ``prototypes`` accessor
returns raw weights for softmax and normalized centers for cosine heads.

QVector and its ResNet and Hugging Face children accept a nested
``qmatrix_code_rate`` dictionary instead of the removed ``qmatrix_code_rate_eps``
argument. ``enable_qmatrix_code_rate`` controls calculation. This loss forces
``reduction="mean"`` and ``distributed_mode="global_data"``. For a q-matrix
[B, num_queries, D], the code-rate implementation evaluates each example across
its queries and averages over B; distributed aggregation applies only to
rank-two inputs. Sample-weighted averages of chunk losses preserve that result.
Unlike SIGReg across xvectors, no global embedding-distribution recomputation
is required for q-matrix code rate.

Training objective and configuration
------------------------------------

For XVectorP, the trainer composes the available terms as

.. math::

   L = L_{\mathrm{head}}
       - \lambda_{\mathrm{rate}} R_{\mathrm{prototypes}}
       + \lambda_{\mathrm{proto}} L_{\mathrm{SIGReg,prototypes}}
       + \lambda_{\mathrm{xvec}} L_{\mathrm{SIGReg,xvectors}}.

Code rate is maximized, hence its negative sign; SIGReg is minimized.
``QVectorTrainer`` additionally subtracts the weighted q-matrix code rate and
supports the prototype terms, but does not calculate xvector SIGReg.
All regularizer weights default to zero. Enable flags independently control
calculation, and each available regularizer value is logged separately.

For example, the model and trainer sections can include:

.. code-block:: yaml

   model:
     enable_xvector_sig_reg: true
     xvector_sig_reg:
       num_slices: 1024
       projection_chunk_size: 256
     head:
       enable_prototype_sig_reg: true
       prototype_sig_reg:
         num_slices: 256
       enable_prototype_code_rate: true
       prototype_code_rate:
         eps: 0.5
   trainer:
     xvector_sig_reg_weight: 0.01
     prototype_sig_reg_weight: 0.01
     prototype_code_rate_weight: 0.01

These model options propagate to ``ResNetXVectorP``, ``HFWav2XVectorP``, and
``HFWav2Vec2XVectorP``. Forced distributed modes and prototype code-rate
normalization override conflicting dictionary values. The standalone loss
classes remain configurable. ``add_class_args`` supports ``skip`` for omitting
constructor argument names from parsers.

``XVectorP.forward`` returns the statistic in ``XVectorPOutput.xvector_sig_reg``;
the trainer owns neither a SIGReg instance nor a cached batch statistic.
``compute_xvector_sig_reg=False`` defers its calculation for internal chunk
forwards. ``infer`` computes it once on the final aggregated embeddings when
enabled. Global statistics require every distributed rank to participate in
the final call. Sampling counters are model buffers and are saved and restored
with model checkpoints.
