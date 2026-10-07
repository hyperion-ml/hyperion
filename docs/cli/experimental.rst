Experimental CLI Commands
=========================

This page owns the experimental command classifications from
``docs/cli_inventory.json``. These commands are documented so they are visible,
not because they carry stable configuration, checkpoint, or output guarantees.

Codec and tokenizer
-------------------

``hyperion-train-dac``, ``hyperion-finetune-dac``, and
``hyperion-train-tokenizer`` require PyTorch and codec-specific data/checkpoints.
Treat encoded-stream formats and checkpoints as version-coupled.

VITS anonymization and voice conversion
---------------------------------------

``hyperion-train-freevc``, ``hyperion-train-vi-anonymizer``,
``hyperion-train-vi-emo-normalizer``, and ``hyperion-finetune-vi-anonymizer``
need the corresponding VITS/FreeVC assets. Validate outputs and privacy/utility
metrics on your own protocol after every upgrade.

Transducers, Q-vectors, and X-vector plus
-----------------------------------------

Transducer training, fine-tuning, and decoding commands require compatible
transducer checkpoints; Wav2Vec2 variants also need ``transformers``. Q-vector
training, fine-tuning, and inference require matching Q-vector checkpoints.
X-vector plus training and inference use ``XVectorPTrainer`` and XVectorP
checkpoints; Hugging Face Wav2Vec2 backbones also need ``transformers``. These
models and checkpoints are not interchangeable with stable x-vector or Q-vector
models. Run ``--help`` for the installed parser and pin the complete
configuration with each experiment.

For example, train a ResNet-backed X-vector plus model with a configuration
file containing the model, training/validation datasets, and trainer settings:

.. code-block:: bash

   hyperion-train-xvectorp resnet --cfg xvectorp.yaml

Inference reads a waveform recording table or a ``HyperDataset`` manifest and
writes embeddings and, optionally, classifier logits. Supply the checkpoint
and at least one output path:

.. code-block:: bash

   hyperion-infer-xvectorps --model-path model.pt --recordings-file wav.scp \
      --xvector-path ark:xvectors.ark

See also
--------

* :doc:`../experimental-components`
* :doc:`../documentation-policy`
