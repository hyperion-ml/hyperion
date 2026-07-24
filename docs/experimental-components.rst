Experimental Components
=======================

The components on this page are part of the maintained package, but they do
not carry the compatibility guarantees of the stable APIs. Their Python
interfaces, configuration schema, training semantics, checkpoint layout, and
output behavior may change in a minor release. Pin the Hyperion revision and
preserve the full configuration, dependency versions, and a small inference
regression test with every trained artifact.

This page intentionally does not provide recipe instructions from ``egs/``.
Use the installed command's ``--help`` output and a version-controlled
``jsonargparse`` configuration as the source of truth for the version you run.

Neural codecs (DAC)
-------------------

The neural codec family is implemented in ``hyperion.torch.models.dac`` and
uses ``DACTrainer``. Both standard and streaming variants are experimental.

* ``hyperion-train-dac``
* ``hyperion-finetune-dac``

These commands use the current ``AudioDataset`` and ``SegSamplerFactory``
configuration contract. Training can require a discriminator and audio-feature
loss configuration in addition to the codec model. Treat encoded streams and
decoder checkpoints as version-coupled: verify encode/decode behavior after an
upgrade before relying on saved artifacts.

VITS anonymization and voice conversion
---------------------------------------

``hyperion.torch.models.freevc`` provides FreeVC and Hugging Face WavLM-based
voice-conversion models. VI anonymizer workflows build on this family and an
audio discriminator; their trainers are ``FreeVCTrainer`` and
``VIAnonymizerTrainer``.

* ``hyperion-train-freevc``
* ``hyperion-train-vi-anonymizer``
* ``hyperion-finetune-vi-anonymizer``

These workflows may require externally sourced pretrained models and substantial
GPU memory. Keep the pretrained model identifier or local revision, sample
frequency, speaker-conditioning setup, and privacy/quality evaluation protocol
beside each checkpoint. Anonymization claims require task-specific evaluation;
a successful training run alone is not evidence of privacy protection.

Transducers
-----------

The transducer families are in ``hyperion.torch.models.transducer`` and
``hyperion.torch.models.wav2transducer``. Their training code uses
``TransducerTrainer``. Some maintained paths depend on ``k2`` and text/token
metadata supplied through the audio dataset configuration.

* ``hyperion-train-wav2rnn-transducer``
* ``hyperion-train-wav2vec2rnn-transducer``
* ``hyperion-train-wav2vec2transducer``
* ``hyperion-finetune-wav2vec2transducer``
* ``hyperion-decode-wav2transducer``
* ``hyperion-decode-wav2vec2rnn-transducer``

Record the tokenizer model and vocabulary, blank/special-token convention,
decoder settings, and exact ``k2`` version with every checkpoint. The decoder
commands are available for inspection but have not yet received a maintained
task tutorial; validate their input and output contract on fixture-scale audio
before a full run.

Q-vectors
---------

Q-vector models and wrappers are located in ``hyperion.torch.models.qvectors``
and use ``QVectorTrainer``. They are not interchangeable with the stable
x-vector extraction and scoring interfaces.

* ``hyperion-train-qvector``
* ``hyperion-finetune-qvector``
* ``hyperion-infer-qvectors``

Version the quantizer/head configuration and all upstream acoustic or Hugging
Face model revisions with the checkpoint. Check output tensor layout and the
meaning of inferred codes for the selected model rather than assuming an
x-vector-compatible embedding matrix.

Adoption checklist
------------------

Before using an experimental component in a long-lived system:

* pin the Hyperion commit or released package version;
* retain the complete YAML/JSON configuration and optional dependency versions;
* save a known input/output regression fixture alongside the model;
* test saving, loading, and inference after every upgrade; and
* explicitly validate task outcomes such as recognition, intelligibility,
  codec fidelity, or anonymization privacy.

See also
--------

* :doc:`documentation-policy`
* :doc:`torch-extension-points`
* :doc:`how-to/use-configuration-files`
* :doc:`how-to/save-load-models-and-backends`
