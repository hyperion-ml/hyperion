PyTorch API Contracts
=====================

This page is the contract-level companion to :doc:`torch-api` and
:doc:`torch-extension-points`. It describes the stable PyTorch boundaries that
callers and extension authors must preserve. Codec/DAC, VITS anonymization,
transducer, and Q-vector model families remain experimental unless explicitly
marked otherwise.

Model serialization and modes
-----------------------------

``HyperTorchModel``
~~~~~~~~~~~~~~~~~~~

:class:`hyperion.torch.HyperTorchModel` is the trainable-model base. A model
subclass must provide a serializable configuration through ``get_config()`` and
should register a stable class name for ``save``, ``load``, and ``auto_load``.

``save(path, ...)`` writes configuration and parameter state; it mutates the
target checkpoint path. ``load(path, ...)`` restores a known class, while
``auto_load(path, ...)`` uses the saved class metadata. A checkpoint is only
compatible when the recorded class, constructor configuration, parameter
shapes, and supported serialization format agree with the installed code.
Changing a model's architecture, classifier vocabulary, embedding dimension,
or frontend configuration can invalidate old checkpoints.

``set_train_mode(mode)`` selects a supported task-specific training mode;
``valid_train_modes`` reports the accepted names. This is separate from
``torch.nn.Module.train()`` / ``eval()``. Use ``train()`` and ``eval()`` to
control dropout, batch normalization, and other runtime behavior; use the
Hyperion train mode only when the model documents distinct optimization modes.

Waveform embedding models
-------------------------

``Wav2XVector`` and ``HFWav2XVector``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.torch.models.wav2xvectors.wav2xvector.Wav2XVector` and
:class:`hyperion.torch.models.wav2xvectors.hf_wav2xvector.HFWav2XVector` are
stable waveform-to-embedding wrappers. They expose ``sample_frequency`` and
``embed_dim``; callers must resample audio to the documented sample frequency
before model invocation.

Waveform batches are floating-point tensors, normally shaped
``(batch, num_samples)`` or ``(batch, channels, num_samples)`` according to the
selected frontend. The associated length/mask input identifies valid samples
for each batch item. Padding must be excluded from temporal pooling; a mask has
the same batch axis and refers to time/sample positions, not embedding axes.

``extract_embed(...)`` runs the embedding extraction path and returns one
embedding per input item, conventionally shaped ``(batch, embed_dim)``. It is
an inference-oriented operation: call ``model.eval()`` and use
``torch.inference_mode()`` or ``torch.no_grad()`` unless gradients are required.
Move both model and all tensor inputs to the same device; CPU/GPU mismatches
raise PyTorch runtime errors rather than triggering implicit transfer.

``Wav2XVector.forward`` accepts ``x`` as a floating-point waveform tensor of
shape ``(batch, num_samples)`` and optional ``x_lengths`` as an integral tensor
of shape ``(batch,)``. Lengths describe valid samples before right padding and
must align with ``x``'s batch axis. ``y`` supplies one target class per batch
item for classification training. With only ordinary classification output
requested, the wrapped backend normally returns logits with shape
``(batch, num_classes)``; requesting encoder/classifier layers can instead
return a structured backend output.

``vad_samples`` and ``vad_feats`` are distinct optional voice-selection inputs:
the former indexes waveform time before feature extraction and the latter
indexes frontend frame time. They must not be reused across those resolutions.
The method forwards frontend features and valid feature lengths to the x-vector
backend. ``train()`` enables training behavior in the frontend/backend,
including dropout and batch-normalization updates; ``eval()`` changes that
module behavior but does not disable gradients by itself.

``Wav2XVector.extract_embed`` accepts the same waveform/length/VAD inputs plus
an optional chunk length in seconds and embedding-layer selector. It returns an
utterance embedding tensor shaped ``(batch, embed_dim)``. Chunking changes
memory use and temporal processing; use the same policy for comparable
enrollment and test embeddings. ``detach_chunks=True`` prevents gradients from
flowing through chunk results.

``HFWav2XVector.forward`` has the same ``x``/``x_lengths`` contract and adds
``return_feat_layers`` for Hugging Face hidden states. It returns logits of
shape ``(batch, num_classes)`` in the simple case, or a structured output with
``logits``, ``h_enc``, ``h_classif``, and optional ``h_feats`` when layer
returns are requested. Its extraction method returns ``(batch, embed_dim)``
and has separate HF-frontend and x-vector-backend chunk-length controls. HF
model weights, feature-processor configuration, selected layers, and backend
configuration are all checkpoint-compatibility inputs.

``NetArch`` and forward contracts
---------------------------------

:class:`hyperion.torch.narchs.net_arch.NetArch` is the neural-architecture
base beneath task models. An architecture's ``forward`` contract must document:

* accepted tensor layout and dtype;
* optional mask/length layout and polarity (``True``/``1`` means valid or
  padded);
* output tensors, their shape, and whether they represent frames, logits, or
  embeddings;
* behavior in ``train()`` versus ``eval()``;
* any state/cache mutation for streaming or recurrent use.

Layers and blocks should not silently transpose time/channel axes or infer a
mask convention from shape alone. A reusable component that changes temporal
resolution must state how it transforms lengths/masks. See
:doc:`torch-layers-and-architectures` for the stable frontend, pooling, block,
and architecture families.

Data and samplers
-----------------

``AudioDataset`` and ``LegacyAudioDataset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.torch.data.audio_dataset.AudioDataset` and
:class:`hyperion.torch.data.legacy_audio_dataset.LegacyAudioDataset` read CSV
recording/segment manifests and return audio plus requested metadata/targets.
They do not define distributed sharding or batch composition; that is the
sampler's responsibility. Dataset ids must match the manifest ids used by
trials and class tables.

Audio chunks have variable duration. A collate function/sampler must either
pad them and supply valid lengths/masks, or form duration-constrained batches.
The batch's target values must remain row-aligned with the audio tensor after
sampling, shuffling, and distributed partitioning.

``AudioDataset.__getitem__`` returns a dictionary containing at least ``id``,
``sample_freq``, and one-dimensional ``audio``. Optional augmentation and
segment attributes are added under configured keys. A string selects a manifest
segment; ``(segment_id, start, duration)`` selects a range. Empty audio is an
error, and unreadable storage/timing propagates a reader error.

``AudioDataset.collate`` sorts examples by decreasing audio length, right-pads
``audio``, and returns ``audio_lengths`` with shape ``(batch,)``. Numeric
targets become tensors, ids remain strings, and other variable-length tensors
receive matching ``<field>_lengths``. Every field is reordered with audio, so
custom training code must consume collated—not pre-sort—target order.
``LegacyAudioDataset`` follows the same rules but uses legacy ``seg_id`` and
usually ``x``/``x_lengths`` keys; do not mix the two key conventions.

``HyperSampler`` and concrete samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.torch.data.hyper_sampler.HyperSampler` is the distributed,
resumable sampler base. ``set_epoch(epoch, batch=0)`` controls deterministic
epoch seeding and resume position. In distributed mode, sampler batch limits
are divided across ranks so that every rank can emit compatible batches.

:class:`hyperion.torch.data.seg_sampler.SegSampler` produces segment-id batches
and respects fixed-size or duration-constrained limits. Factory methods such as
``SegSamplerFactory.create`` choose a concrete sampler from configuration.
Calling a factory with an unknown sampler type, incompatible base sampler, or
impossible per-rank batch budget raises :class:`ValueError`.

Samplers yield ids or id/start/duration tuples, not tensors. ``set_epoch``
mutates deterministic seed and resume state. For exact resume, restore both the
trainer checkpoint and sampler epoch/batch state before creating the iterator.

Trainer and checkpoint contracts
--------------------------------

``TorchTrainerBase``
~~~~~~~~~~~~~~~~~~~~

:class:`hyperion.torch.trainers.torch_trainer_base.TorchTrainerBase` owns the
optimization loop, checkpoint lifecycle, logging, gradient accumulation,
validation scheduling, AMP, distributed execution, scheduler updates, and
optional stochastic weight averaging. Task-specific trainers supply the model
and batch/metric behavior; they should not reimplement checkpoint semantics.

``save_checkpoint()`` persists model, optimizer, scheduler, training progress,
and other resumable state. ``load_last_checkpoint()`` restores the most recent
compatible state. Resume correctness requires the same model configuration,
optimizer/scheduler family, class vocabulary, and distributed topology policy.
Changing world size or FSDP/DDP policy may require an explicitly supported
conversion path rather than ordinary resume.

Modern subclasses write a complete checkpoint directory per epoch/step, with
shared ``trainer_state.json`` and ``rng_state.pth`` at its root and one
subdirectory for each model. Model configuration is ``config.json`` and model
weights are ``model.safetensors``; optimizer and scheduler states are separate
``.pt`` files. The directory is written temporarily and renamed only when
complete, and discovery requires shared state plus every required model's
configuration, weights, and optimizer state. ``LegacyTorchTrainer`` retains
its separate legacy ``.pth`` behavior. See
:doc:`how-to/manage-torch-checkpoints` for the layout and migration command.

With ordinary DDP, rank 0 writes the shared checkpoint and every rank restores
it on resume. With FSDP, every rank participates in the full-state collection
and restore collectives, while only rank 0 publishes files to disk. This is
required to keep FSDP model and optimizer state consistent across ranks.

``use_amp`` and ``amp_dtype`` control mixed precision. They affect numerical
execution but do not automatically make unsupported operations safe. Gradient
scaling, clipping, and accumulation occur in the trainer's prescribed order;
custom trainers must preserve that order. All ranks must follow equivalent
control flow for distributed training and validation.

``validation_step(batch_idx, batch_data)`` preprocesses a collated batch, moves
supported tensors to the configured device, executes the model, and returns
``(batch_size, metrics)``. A task trainer must document its required batch keys,
layouts, target dtype, and metric values. Non-tensor metadata such as ids stays
on the host unless a preprocessing hook uses it.

``load_last_checkpoint()`` returns saved log metadata or ``None`` when no
checkpoint exists; when one exists it mutates trainer, model, optimizer, and
scheduler state. It is for training resume, not for loading a standalone
deployment model.

Factories and configuration
---------------------------

``OptimizerFactory``, ``LRSchedulerFactory``, and ``WDSchedulerFactory``
construct training components from configuration. Their ``filter_args`` methods
select only supported configuration keys, while ``add_class_args`` exposes
parser options through ``jsonargparse``. Store the resulting model, trainer,
optimizer, and scheduler configuration together with a checkpoint; a state
dictionary alone is insufficient to reproduce an experiment.

Focused inference example
--------------------------

.. code-block:: python

   import torch
   from hyperion.torch.models.wav2xvectors.wav2xvector import Wav2XVector

   model = Wav2XVector.auto_load("exp/model/checkpoint.pt")
   model.eval()
   wav = torch.randn(2, 16_000, device="cuda")
   lengths = torch.tensor([16_000, 12_000], device="cuda")
   model = model.to(wav.device)
   with torch.inference_mode():
       embeddings = model.extract_embed(wav, x_lengths=lengths)
   assert embeddings.shape == (2, model.embed_dim)

The exact argument name for lengths/masks is model-specific; consult the
generated class signature before using this example with a different wrapper.

See also
--------

* :doc:`torch-api`
* :doc:`torch-extension-points`
* :doc:`torch-layers-and-architectures`
* :doc:`torch-training-support`
* :doc:`how-to/train-waveform-xvector`
* :doc:`how-to/run-resumable-distributed-training`
