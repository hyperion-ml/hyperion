PyTorch API Overview
====================

This page is the curated entry point for Hyperion's PyTorch stack. The detailed
legacy/reference material remains in :doc:`torch`; use this page to locate the
supported extension points and their runtime contracts.

Model contract
--------------

.. autoclass:: hyperion.torch.HyperTorchModel
   :no-index:
   :members: get_config, save, load, auto_load, set_train_mode, valid_train_modes

Trainable models derive from ``HyperTorchModel`` and serialize their class
configuration with parameters. Inputs, output shape, and mask conventions are
model-specific; waveform x-vector wrappers accept waveform batches and expose
their expected sample frequency and embedding dimension.

.. autoclass:: hyperion.torch.models.wav2xvectors.wav2xvector.Wav2XVector
   :no-index:
   :members: sample_frequency, embed_dim, extract_embed

Data and sampling
-----------------

.. autoclass:: hyperion.torch.data.legacy_audio_dataset.LegacyAudioDataset
   :no-index:

``LegacyAudioDataset`` reads CSV recording/segment manifests and class CSVs.
Samplers determine chunk and batch duration, which is the principal memory
control for waveform training. See :doc:`how-to/train-waveform-xvector`.

Training
--------

.. autoclass:: hyperion.torch.trainers.torch_trainer_base.TorchTrainerBase
   :no-index:
   :members: load_last_checkpoint, save_checkpoint, add_class_args

The trainer owns checkpointing, logging, AMP, DDP/FSDP policy, gradient
accumulation, schedulers, and validation cadence. Configure it through the
``trainer`` mapping in a command config; see
:doc:`how-to/run-resumable-distributed-training`.

Architecture layers
-------------------

The PyTorch package is intentionally layered:

* ``layers``: primitive operations;
* ``layer_blocks``: reusable compositions;
* ``narchs``: neural architectures;
* ``models``: task-level models;
* ``trainers`` and ``data``: training execution.

New task models belong in ``models`` and should compose documented
architectures/blocks rather than duplicating trainer or data-loader behavior.
For the architecture, model, data, sampler, trainer, and factory contracts,
see :doc:`torch-extension-points`.

For selecting stable feature frontends, pooling, reusable blocks, and neural
architecture families, see :doc:`torch-layers-and-architectures`.


V2 Transformer references
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.transformer_encoder_v2.TransformerEncoderV2
   :no-index:
   :members: forward, init_state, out_shape, get_config, requires_ddp_find_unused_parameters

.. autoclass:: hyperion.torch.narchs.qformer_v2.QFormerV2
   :no-index:
   :members: forward, out_shape, get_config, requires_ddp_find_unused_parameters

.. autoclass:: hyperion.torch.layer_blocks.transformer_v2.TransformerV2G4MoEBlock
   :no-index:
   :members: forward

.. autoclass:: hyperion.torch.layers.attention_v2.HFFlashScaledDotProdAttV2
   :no-index:
   :members: compute_attention

.. autoclass:: hyperion.torch.layers.norm_layers.RMSNorm
   :no-index:
   :members: forward

For shapes, length transformations, external cache mutation, and checkpoint
migration, see :doc:`torch-api-contracts`. For configuration defaults,
normalization, MoE routing, backend prerequisites, and a synthetic encoder to
QFormer workflow, see :doc:`torch-layers-and-architectures`.

Margin-based classifier heads, training metrics/loggers, and resumable
scheduler behavior are documented in :doc:`torch-training-support`.

Experimental model families
---------------------------

Codec/DAC, VITS anonymization, transducer, q-vector, and X-vector plus models
are experimental. TPM wrappers and adversarial modules are stable but may require
external model packages. See :doc:`documentation-policy` before choosing an
extension or deployment target.

Third-party integrations and adversarial robustness
----------------------------------------------------

Hugging Face frontends, DNSMOS/UTMOS/VoxProfile evaluators, and the adversarial
attack/defense interfaces are stable PyTorch surfaces with external runtime or
model-asset requirements. Their contracts and reproducibility requirements are
documented in :doc:`torch-integrations-and-robustness`.

See also
--------

* :doc:`torch`
* :doc:`torch-extension-points`
* :doc:`torch-layers-and-architectures`
* :doc:`torch-training-support`
* :doc:`torch-integrations-and-robustness`
* :doc:`how-to/train-waveform-xvector`
* :doc:`how-to/train-pretrained-wav2vec2-xvector`
* :doc:`how-to/save-load-models-and-backends`
