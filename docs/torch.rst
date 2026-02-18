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

Generic torch model loader:

.. autoclass:: hyperion.torch.TorchModelLoader

Neural architecture base class:

.. autoclass:: hyperion.torch.narchs.net_arch.NetArch

Architecture loader:

.. autoclass:: hyperion.torch.narchs.torch_na_loader.TorchNALoader

Layering model
--------------

The stack follows this composition flow:

1. ``hyperion.torch.layers`` (primitive operations)
2. ``hyperion.torch.layer_blocks`` (composite blocks)
3. ``hyperion.torch.narchs`` (architectures)
4. ``hyperion.torch.models`` (task models)

Layer and block namespaces
--------------------------

.. automodule:: hyperion.torch.layers
   :members:

.. automodule:: hyperion.torch.layer_blocks
   :members:

Neural architectures
--------------------

``hyperion.torch.narchs`` includes architecture families used by models, such as
ResNet/Res2Net variants, conformer/transformer encoders, ConvNeXt encoders,
DAC encoder/decoder, QFormer/Hydra heads, and auxiliary heads.

.. automodule:: hyperion.torch.narchs
   :members:

Top-level models
----------------

``hyperion.torch.models`` contains end-to-end models used by scripts and trainers,
including x-vector, wav2xvector, qvector, DAC, FreeVC, transducer, and VAE-related
models.

``wav2qvectors`` and ``fa_codec`` are currently not documented.

.. automodule:: hyperion.torch.models
   :members:

Data pipeline and samplers
--------------------------

Main dataset and sampler entry points are exported in ``hyperion.torch.data``.

.. automodule:: hyperion.torch.data
   :members:

Learning rate and weight decay schedulers
-----------------------------------------

.. automodule:: hyperion.torch.lr_schedulers
   :members:

.. automodule:: hyperion.torch.wd_schedulers
   :members:

Training stack
--------------

Canonical trainer base classes:

.. autoclass:: hyperion.torch.trainers.torch_trainer_base.TorchTrainerBase

.. autoclass:: hyperion.torch.trainers.single_model_trainer.SingleModelTrainer

Current trainers built on canonical stack and actively documented:

.. autoclass:: hyperion.torch.trainers.qvector_trainer.QVectorTrainer

.. autoclass:: hyperion.torch.trainers.dac_trainer.DACTrainer

.. autoclass:: hyperion.torch.trainers.freevc_trainer.FreeVCTrainer

.. autoclass:: hyperion.torch.trainers.vi_anonymizer_trainer.VIAnonymizerTrainer

Legacy trainer note
~~~~~~~~~~~~~~~~~~~

``LegacyTorchTrainer`` remains available and is still used for x-vector training
flows.

.. autoclass:: hyperion.torch.trainers.legacy_torch_trainer.LegacyTorchTrainer

.. autoclass:: hyperion.torch.trainers.xvector_trainer.XVectorTrainer

For now, other legacy trainers (for example transducer/VAE/DVAE legacy paths) are
intentionally not documented here.

Torch metrics
-------------

Torch metric utilities are in ``hyperion.torch.metrics``.

.. automodule:: hyperion.torch.metrics
   :members:

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
