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

Projection weight decay is configured through ``proj_weight_decay`` or the
``--model.proj-weight-decay`` CLI option for training and fine-tuning.

* ``hyperion-train-qvector``
* ``hyperion-finetune-qvector``
* ``hyperion-infer-qvectors``

Version the quantizer/head configuration and all upstream acoustic or Hugging
Face model revisions with the checkpoint. Check output tensor layout and the
meaning of inferred codes for the selected model rather than assuming an
x-vector-compatible embedding matrix.

X-vector plus
-------------

XVectorP models add global pooling and a projection head to backbone features,
with an optional classification or regression head. They are trained with
``XVectorPTrainer``.

* ``hyperion-train-xvectorp``
* ``hyperion-infer-xvectorps``

``hyperion.torch.models.xvectorps`` provides the experimental ``XVectorP`` base
class and waveform-based ``ResNetXVectorP``. Hugging Face waveform wrappers
``HFWav2XVectorP`` and ``HFWav2Vec2XVectorP`` are also in
``hyperion.torch.models.xvectorps``. These models pool the final backbone
features globally, project the pooled vector with ``ProjHead``, and pass the
embedding to a Hydra classification or regression head. Backbone features use
``(batch, time, features)``; pooling receives ``(batch, features, time)`` and
valid frame lengths. The Hugging Face wrappers use the final hidden state by
default and can optionally fuse a selected range of hidden states first.

The projection options are ``proj_use_norm``, ``proj_norm_layer`` (``batch-norm``,
``layer-norm``, or ``rms-norm``), and ``proj_norm_before``. Normalization is enabled
before projection by default, with batch normalization selected when the type
is omitted. Projection bias is enabled with input normalization or disabled
normalization, and disabled with output normalization. These options also have
fine-tuning overrides; an omitted override keeps the checkpoint setting.
Changing projection structure rebuilds the projection, while retaining compatible
linear weights when the embedding dimension is unchanged.

Pooling configuration lives under ``pooling``. Component weight-decay overrides
are ``pooling_weight_decay``, ``proj_weight_decay``, and ``head_weight_decay``;
``bias_weight_decay`` also applies to normalization parameters. Training modes
are ``full``, ``frozen``, ``frozen-feat-extractor``, ``pooling``, ``proj-head``, and
``output-layer``. Inference averages chunk embeddings using valid sample counts.
``XVectorPOutput`` contains the embedding, optional head output, optional
backbone features, and optional ``xvector_sig_reg``. Xvector and prototype
SIGReg options, code-rate configuration dictionaries, trainer weights,
and fine-tuning overrides are documented in :doc:`sig-reg`.

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
