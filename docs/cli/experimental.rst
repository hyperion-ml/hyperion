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

Transducers and Q-vectors
-------------------------

Transducer training, fine-tuning, and decoding commands require compatible
transducer checkpoints; Wav2Vec2 variants also need ``transformers``. Q-vector
training, fine-tuning, and inference require matching Q-vector checkpoints.
Run ``--help`` for the installed parser and pin the complete configuration with
each experiment.

See also
--------

* :doc:`../experimental-components`
* :doc:`../documentation-policy`
