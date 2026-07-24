PyTorch Stack
=============

Overview
--------

The ``hyperion.torch`` package contains:

* low-level layers and layer blocks
* neural architectures (``narchs``)
* top-level models
* data pipelines, samplers, trainers, schedulers
* third-party model wrappers in ``hyperion.torch.tpm``

Core model abstractions
-----------------------

All torch models derive from:

.. autoclass:: hyperion.torch.HyperTorchModel
   :members: get_config, save, load, auto_load, set_train_mode, valid_train_modes

Generic torch model loader:

.. autoclass:: hyperion.torch.TorchModelLoader
   :members: load

Neural architecture base class:

.. autoclass:: hyperion.torch.narchs.net_arch.NetArch
   :members: in_context, in_dim, out_dim, in_shape, out_shape

Architecture loader:

.. autoclass:: hyperion.torch.narchs.torch_na_loader.TorchNALoader
   :members: load

Layering model
--------------

The stack follows this composition flow:

1. ``hyperion.torch.layers`` (primitive operations)
2. ``hyperion.torch.layer_blocks`` (composite blocks)
3. ``hyperion.torch.narchs`` (architectures)
4. ``hyperion.torch.models`` (task models)

Layer and block namespaces
--------------------------

``layers`` contains primitive operations such as feature frontends, pooling,
normalization, margin heads, and vector quantization. ``layer_blocks`` composes
those primitives into reusable encoder/decoder and residual building blocks.
Their public forward contracts must state tensor layout, valid-length/mask
handling, output layout, and train/eval behavior; implementation-private blocks
are intentionally not listed as a supported API.

See :doc:`torch-layers-and-architectures` for supported factories and component
families, and :doc:`torch-api-contracts` for the required forward-contract and
device semantics.

Neural architectures
--------------------

``hyperion.torch.narchs`` includes architecture families used by models, such as
ResNet/Res2Net variants, conformer/transformer encoders, ConvNeXt encoders,
DAC encoder/decoder, QFormer/Hydra heads, and auxiliary heads.

Use the architecture factories for supported ResNet, TDNN, SpineNet, and related
families. ``NetArch`` implementations are reusable neural architectures, not
end-user task models: their input/output tensor and mask contracts must be
preserved by the enclosing model.

See :doc:`torch-layers-and-architectures` and :doc:`torch-api-contracts`.

Top-level models
----------------

``hyperion.torch.models`` contains end-to-end models used by scripts and trainers,
including x-vector, wav2xvector, qvector, DAC, FreeVC, transducer, and VAE-related
models.

Stable x-vector and waveform x-vector models are documented through
:doc:`torch-api` and :doc:`torch-api-contracts`. Codec/DAC, VITS/freevc,
transducer, and Q-vector model families are experimental and are covered only
in :doc:`experimental-components`.

Data pipeline and samplers
--------------------------

The stable public entry points are ``AudioDataset``, ``LegacyAudioDataset``,
``HyperSampler``, ``SegSampler``, and their factories. Dataset outputs, padded
batch layout, length fields, deterministic sampler state, and distributed
resume behavior are documented in :doc:`torch-api-contracts`.

Detailed sampler documentation:

.. toctree::
   :maxdepth: 2

   torch/data/datasets
   torch/data/seg_samplers
   torch/data/embed_samplers

Learning rate and weight decay schedulers
-----------------------------------------

Use ``LRSchedulerFactory`` and ``WDSchedulerFactory`` through trainer
configuration rather than importing scheduler implementations directly. Their
state dictionaries belong to a resumable trainer checkpoint and must be loaded
with a compatible optimizer/trainer configuration. See
:doc:`torch-training-support`.

Training stack
--------------

Canonical trainer base classes:

.. autoclass:: hyperion.torch.trainers.torch_trainer_base.TorchTrainerBase

.. autoclass:: hyperion.torch.trainers.single_model_trainer.SingleModelTrainer

Experimental trainer families (Q-vector, DAC, FreeVC, and VITS anonymization)
are intentionally not expanded here as a stable reference. Their current scope
and compatibility caveats are recorded in :doc:`experimental-components`.

Legacy trainer note
~~~~~~~~~~~~~~~~~~~

``LegacyTorchTrainer`` remains available and is still used for x-vector training
flows.

.. autoclass:: hyperion.torch.trainers.legacy_torch_trainer.LegacyTorchTrainer
   :members: save_checkpoint, load_checkpoint

.. autoclass:: hyperion.torch.trainers.xvector_trainer.XVectorTrainer
   :members: train_epoch, validation_epoch

For now, other legacy trainers (for example transducer/VAE/DVAE legacy paths) are
intentionally not documented here.

Torch metrics
-------------

Torch metric utilities support training-loop aggregation and are distinct from
trial-based speaker-verification metrics. Use the latter for EER/DCF reporting;
see :doc:`metrics`. Training metric/logger and scheduler hook behavior is
documented in :doc:`torch-training-support`.

Third-party model wrappers (TPM)
--------------------------------

``hyperion.torch.tpm`` is a first-class subsystem for wrappers around external
models and toolkits.

Wrapper families:

* ``hyperion.torch.tpm.hf``: Hugging Face wrappers
  ``HFWav2Vec2``, ``HFHubert``, ``HFWavLM``, ``WhisperTranscriber``.
* ``hyperion.torch.tpm.dnsmos``: ``DNSMOS`` speech quality wrapper.
* ``hyperion.torch.tpm.utmos``: ``UTMOSV2`` wrapper.
* ``hyperion.torch.tpm.usc``: VoxProfile evaluators.

Model/checkpoint behavior
~~~~~~~~~~~~~~~~~~~~~~~~~

These wrappers are designed to download pretrained models/checkpoints
automatically when needed.

VoxProfile dependency
~~~~~~~~~~~~~~~~~~~~~

VoxProfile wrappers require installing Hyperion with the ``voxprofile`` extra:

.. code-block:: bash

   pip install -e .[voxprofile]

Combined example:

.. code-block:: bash

   pip install -e .[torch29,voxprofile]
