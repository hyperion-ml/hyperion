PyTorch Extension Workflows
===========================

This guide explains how to add maintained PyTorch functionality without
duplicating model, trainer, or command-line infrastructure. It complements
:doc:`model-extension-contracts`: use that page for the serialization and
checkpoint contract, and this page for choosing and implementing the right
PyTorch extension point.

Choose the lowest appropriate layer
-----------------------------------

Build from the inside out. The dependency direction is deliberate:

``layers`` → ``layer_blocks`` → ``narchs`` → ``models`` → ``trainers``.

.. list-table:: Where a new behavior belongs
   :widths: 27 33 40
   :header-rows: 1

   * - New behavior
     - Home
     - Contract to preserve
   * - A small reusable operation
     - ``hyperion.torch.layers``
     - A normal ``torch.nn.Module`` forward contract.
   * - A reusable composition of operations
     - ``hyperion.torch.layer_blocks``
     - Component shape, dtype, and mask/length behavior.
   * - An encoder, decoder, or other reusable network
     - ``hyperion.torch.narchs``
     - ``NetArch`` configuration and shape-reporting methods.
   * - Task-specific prediction, embeddings, or loss-facing behavior
     - ``hyperion.torch.models``
     - ``HyperTorchModel`` configuration and checkpoint compatibility.
   * - A new batch structure or sampling policy
     - ``hyperion.torch.data``
     - The selected dataset, sampler, and trainer batch contract.
   * - A genuinely new optimization loop
     - ``hyperion.torch.trainers``
     - Resume, distributed, AMP, logging, and checkpoint behavior.
   * - A general optimizer or schedule choice
     - The corresponding factory
     - Factory selection plus ``jsonargparse`` integration.

Do not put task loss logic in a layer block, and do not put optimizer steps,
distributed wrapping, or checkpoint writing in a model. If a change only
affects one task, it normally belongs in that task's model or trainer, not in a
generic architecture.

Adding a primitive layer
------------------------

Add a primitive only when it will be useful independently of one architecture.
It should derive from ``torch.nn.Module`` and have a narrow, explicit forward
contract.

* State the accepted tensor layout, dtype, output layout, and whether the
  operation preserves the time axis.
* State how padding, masks, sequence lengths, and empty sequences are handled.
  A layer must not silently reinterpret a mask from a caller.
* Keep train/eval differences inside standard PyTorch module behavior, such as
  dropout or batch normalization. Document effects that change numerical
  output between ``train()`` and ``eval()``.
* Register parameters and persistent tensors as modules, parameters, or
  buffers. Do not retain device-specific tensor state in unregistered Python
  attributes; registered state follows ``.to(device)`` and is checkpointed.
* Add a focused unit test for shape, dtype, gradient flow, and any mask or
  length edge case. Test both training and evaluation modes when they differ.

Use a direct ``nn.Module`` rather than ``HyperTorchModel`` for a layer. The
latter is for serializable architecture and task-model objects, not every
reusable operation.

Adding a layer block
--------------------

A layer block is a reusable composition of layers. It also normally derives
directly from ``torch.nn.Module``. Its public constructor should expose the
choices that other architectures genuinely need, while keeping task choices
such as speaker classes, targets, or decoder vocabularies out of the block.

Document the block's input and output tensor layout and every shape-changing
parameter (channels, stride, pooling, context, and projection dimensions).
When a block accepts a mask or lengths, forward that information to each
component that needs it and describe the returned mask/length convention. Add
a test that composes the block with a representative preceding and following
module; this catches channel- and time-axis assumptions that a standalone
shape test can miss.

Adding a ``NetArch``
--------------------

Use ``hyperion.torch.narchs.NetArch`` for a reusable network architecture.
``NetArch`` is a ``HyperTorchModel`` and therefore participates in model
configuration and dynamic loading. It adds the shape-reporting contract:

* ``in_shape()`` returns the expected input shape **including the batch axis**.
* ``out_shape(in_shape=None)`` returns the output shape, also including the
  batch axis. Honor ``in_shape`` when the architecture supports a dynamic input
  shape.
* ``in_context()`` returns the required left/right input context in frames;
  return zero only when the architecture has no extra context requirement.
* ``in_dim()`` and ``out_dim()`` are derived from those shape methods and
  should need no override in ordinary architectures.

Implement ``forward`` with the same layout that ``in_shape`` declares. If the
architecture accepts lengths or masks, make their type, shape, device, and
return behavior explicit in its docstring. A model wrapper cannot infer those
semantics from ``out_shape`` alone.

The architecture constructor must retain enough JSON-friendly state to
recreate itself through ``get_config()``. Import a loader-facing architecture
from the appropriate maintained package initializer so the class is imported
before a serialized configuration is resolved. The registry is populated when
the subclass is imported; merely defining an unimported module does not make a
checkpoint reliably loadable in another process.

Validate a new architecture with a construction/forward test, shape-method
tests, a configuration round trip, and a test using the smallest task model
that consumes it. See :doc:`model-extension-contracts` for the precise
``get_config()``, save, and load requirements.

Adding a task model
-------------------

Use a class in ``hyperion.torch.models`` only when it owns a task-level
interface: for example, embeddings and classification for an x-vector model,
or reconstruction and latent behavior for an autoencoder. Derive from
``HyperTorchModel`` and compose a ``NetArch`` or layer blocks rather than
reimplementing their computation.

Before writing a trainer, inspect the maintained trainer that is closest to
the task. A model compatible with an existing trainer is preferable to a new
trainer. Its public documentation must state:

* the input tensors, layouts, dtypes, device expectations, and mask/length
  representation;
* each forward return value and its shape, including which value is consumed by
  the loss or embedding/scoring path;
* the effects of train modes, ``train()``/``eval()``, freezing, and any
  pretrained component policy; and
* configuration fields, artifacts saved by ``save()``, and which old
  checkpoint/configuration forms remain supported.

Preserve configuration keys and tensor names whenever possible. If a stable
model needs a migration, keep the compatibility logic near loading and add a
test with an old-format fixture. The model extension contract contains the
required ``get_config()``, ``load()``, and ``auto_load()`` behavior.

Adding datasets and samplers
----------------------------

For new waveform-oriented work, prefer ``AudioDataset``. It is the current
dataset interface; ``LegacyAudioDataset`` remains for maintained commands that
use its existing batch format. A dataset change is also a trainer integration
change: inspect the trainer's batch keys before changing what ``__getitem__``
returns.

Use ``SegSamplerFactory`` for a reusable sequential-audio sampler. Register a
new sampler type in its maintained mapping, implement the sampler's argument
filtering/configuration pattern, and expose it through the factory rather than
adding a special case in a command. Ordering must be reproducible and
rank-aware; implement and test ``set_epoch`` where the sampler's random order
depends on the epoch. Treat ``max_batch_length`` as part of the padded-batch
memory contract, not as an optional cosmetic setting.

Test dataset changes with a tiny CSV manifest and a dataloader batch. Assert
the keys, tensor shapes, dtypes, labels, lengths/masks, and a repeated-epoch
sampling result. Use CSV manifests in examples and tests; the retired ``.scp``
format is not a new public interface.

Adding or extending a trainer
-----------------------------

``TorchTrainerBase`` owns generic execution policy: checkpoint scheduling and
resume, gradient accumulation and clipping, AMP, logging, distributed DDP or
FSDP execution, scheduler handling, and optional SWA. ``SingleModelTrainer``
adds the ordinary single-model contract: a ``HyperTorchModel``, optimizer,
optional schedules/loss, and named batch input and target keys.

Create a task-specific trainer only when the model cannot satisfy one of the
existing trainer contracts. Start from the closest maintained trainer and
preserve its lifecycle methods, checkpoint contents, logger event names, and
``add_class_args``/``filter_args`` interface. In particular, a custom loop
must work for CPU/single-process execution before it can be regarded as
compatible with AMP or distributed execution. Do not assume that code running
on one GPU is safe under gradient accumulation, DDP, or FSDP.

For any trainer change, run a short train-and-resume test. It should cover an
optimizer update, checkpoint creation, restore of model/optimizer/scheduler
state, epoch and global-step continuation, and validation logging. Add AMP and
multi-rank coverage when the modification touches those paths.

Adding optimizers, schedules, and losses
-----------------------------------------

Add a general optimizer option through ``OptimizerFactory`` and a reusable
learning-rate or weight-decay policy through ``LRSchedulerFactory`` or
``WDSchedulerFactory``. Each factory is the configuration boundary: update
``create()``, make ``filter_args()`` admit exactly the supported keys, and
extend ``add_class_args()`` so JSON/YAML configuration and CLI help expose the
same choice. Reject unknown identifiers with a clear ``ValueError``.

A loss that is broadly useful belongs in ``hyperion.torch.losses`` and should
be an ``nn.Module`` with documented reduction, target type/shape, numerical
assumptions, and behavior for padded values. Keep a task-specific combination
of losses in the task model or trainer. Never add a CLI-only loss option that
cannot be represented in the saved configuration.

Completion checklist
--------------------

Before submitting a maintained PyTorch extension, verify the items that apply:

* Unit tests cover construction, forward output contracts, gradients, and
  train/eval behavior; mask/length and device cases are included when used.
* A ``NetArch`` reports correct batch-inclusive shapes and round-trips through
  its configuration; a task model saves and reloads a compatible artifact.
* Datasets and samplers produce the batch keys required by the chosen trainer
  and preserve deterministic per-epoch ordering.
* Trainer changes preserve checkpoint/resume and run a minimal CPU training
  cycle; distributed or AMP changes receive targeted coverage.
* Factory and CLI-facing changes use ``add_class_args`` and ``filter_args``;
  config keys are stable and documented.
* Public classes have API-contract documentation, and examples use supported
  CSV inputs. Label experimental codec/DAC, VITS anonymization/freevc,
  transducer, and q-vector interfaces as experimental rather than presenting
  them as stable foundations.

See also
--------

* :doc:`contributor-extension-guide`
* :doc:`model-extension-contracts`
* :doc:`torch-extension-points`
* :doc:`torch-api-contracts`
* :doc:`torch-training-support`
* :doc:`building-documentation`
