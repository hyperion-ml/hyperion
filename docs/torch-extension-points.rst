PyTorch Extension Points
========================

This reference describes the contracts to use when extending Hyperion's
PyTorch stack. It is intentionally selective: the public base classes,
factories, and data interfaces below are the supported places to integrate new
models and training behavior. For a runnable waveform x-vector workflow, see
:doc:`how-to/train-waveform-xvector`.

Layering and ownership
----------------------

Hyperion separates a reusable network from the task that trains it:

* ``layers`` contain primitive operations;
* ``layer_blocks`` compose primitives into reusable modules;
* ``narchs`` contain network architectures and report their tensor shapes;
* ``models`` own task-specific forward, loss, and embedding behavior;
* ``data`` produces batches and sampling plans; and
* ``trainers`` own the optimization loop, checkpoints, logging, AMP, and
  distributed execution.

Add a reusable encoder or decoder to ``narchs``. Add a model only when it
defines task-level behavior. Do not put checkpointing or distributed-launch
logic in either: that responsibility belongs to a trainer.

Neural architectures
--------------------

.. autoclass:: hyperion.torch.narchs.net_arch.NetArch
   :no-index:
   :members: in_context, in_dim, out_dim, in_shape, out_shape

``NetArch`` is the architecture-level base class. Its shape methods describe
tensor shapes *including the batch axis*. Implement them accurately: model
wrappers and configuration validation depend on the reported channel and time
dimensions. ``in_context`` expresses any frame context the architecture needs.

The maintained architecture families include TDNN, ResNet/ResNet1d,
Conformer, Transformer, ConvNeXt, EfficientNet, and SpineNet. Select one that
matches the input representation rather than copying its implementation into a
task model. Architectures that are specific to codec/DAC or transducer systems
are experimental.

Task models and waveform wrappers
---------------------------------

.. autoclass:: hyperion.torch.HyperTorchModel
   :no-index:
   :members: get_config, save, load, auto_load, set_train_mode, valid_train_modes

``HyperTorchModel`` is the serializable base for trainable task models. A
subclass must retain a JSON-friendly configuration and use the inherited
save/load mechanism so that ``auto_load`` can recreate it from an artifact.
The model's ``forward`` contract is task-specific; document its input tensor
layout, length/mask semantics, target fields, and returned values alongside the
model.

.. autoclass:: hyperion.torch.models.wav2xvectors.wav2xvector.Wav2XVector
   :no-index:
   :members: sample_frequency, embed_dim, forward, extract_embed

.. autoclass:: hyperion.torch.models.wav2xvectors.hf_wav2xvector.HFWav2XVector
   :no-index:
   :members: sample_frequency, forward_feats, forward, extract_embed

``Wav2XVector`` accepts waveforms shaped ``(batch, num_samples)``. It performs
toolkit acoustic feature extraction before calling the x-vector backend.
``HFWav2XVector`` follows the same task contract but uses a Hugging Face feature
extractor and fuses selected hidden layers. Its pretrained checkpoint, cache,
and fine-tuning policy should be explicit in the model configuration; see
:doc:`how-to/train-pretrained-wav2vec2-xvector`.

Datasets and samplers
---------------------

.. autoclass:: hyperion.torch.data.audio_dataset.AudioDataset
   :no-index:
   :members: set_epoch, __getitem__

.. autoclass:: hyperion.torch.data.legacy_audio_dataset.LegacyAudioDataset
   :no-index:
   :members: __getitem__

``AudioDataset`` is the current dataset interface for waveform training. It
loads a ``HyperDataset`` and can return class labels, tokenized attributes,
extra metadata, and augmentations. ``LegacyAudioDataset`` is kept for existing
CSV-manifest x-vector workflows; prefer ``AudioDataset`` for new integrations
unless the maintained command you are extending requires the legacy batch
format.

.. autoclass:: hyperion.torch.data.hyper_sampler.HyperSampler
   :no-index:
   :members: set_epoch

.. autoclass:: hyperion.torch.data.seg_sampler.SegSampler
   :no-index:

.. autoclass:: hyperion.torch.data.seg_sampler_factory.SegSamplerFactory
   :no-index:
   :members: create, filter_args, add_class_args

Samplers emit batch index lists and are responsible for reproducible
rank-aware ordering. Call ``set_epoch`` on resume and at every epoch boundary.
Use ``max_batch_length`` to bound padded waveform cost; it is generally a more
reliable memory control than only setting a fixed batch size. The sampler
factory is the public configuration boundary for the sequence sampler family.

Training and optimization
-------------------------

.. autoclass:: hyperion.torch.trainers.torch_trainer_base.TorchTrainerBase
   :no-index:
   :members: load_last_checkpoint, save_checkpoint, add_class_args

.. autoclass:: hyperion.torch.trainers.xvector_trainer.XVectorTrainer
   :no-index:

.. autoclass:: hyperion.torch.trainers.xvector_trainer_from_wav.XVectorTrainerFromWav
   :no-index:

``TorchTrainerBase`` owns generic checkpoint and launch policy. The x-vector
trainers define the speaker-classification batch and loss convention;
``XVectorTrainerFromWav`` additionally applies an acoustic feature extractor.
Choose an existing trainer before creating one: custom models normally only
need to satisfy the selected trainer's model and batch contract. See
:doc:`how-to/run-resumable-distributed-training` for resume, AMP, and DDP.

.. autoclass:: hyperion.torch.optim.factory.OptimizerFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.lr_schedulers.factory.LRSchedulerFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.wd_schedulers.factory.WDSchedulerFactory
   :no-index:
   :members: create, filter_args, add_class_args

These factories are the configuration-facing interface for optimizers and
schedulers. Add a new optimizer or schedule here only when it is intended for
general reuse; a model-specific learning rule belongs with its trainer and
should be documented as such.

Stable and experimental families
--------------------------------

The general layers, layer blocks, architecture families, waveform x-vector
models, datasets/samplers, trainer infrastructure, TPM wrappers, and
adversarial components are stable public surfaces. Codec/DAC, VITS
anonymization and voice conversion, transducers, and q-vectors remain
experimental: their configuration and serialized artifacts may change between
releases. Refer to :doc:`documentation-policy` before building against an
experimental family.

See also
--------

* :doc:`torch-api`
* :doc:`how-to/use-configuration-files`
* :doc:`how-to/save-load-models-and-backends`
