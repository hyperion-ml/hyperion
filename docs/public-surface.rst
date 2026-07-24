Public Surface Inventory
========================

This inventory defines the documentation scope for the current package. It is
an architectural map, not a promise that every implementation module is a
directly supported import path. The API-reference work will use it to decide
which modules require curated pages and which types should be treated as
implementation details.

Stable package areas
--------------------

``hyperion.io``
  Audio, feature, HDF5, ARK, Kaldi-style, and VAD readers/writers. The public
  reference will cover reader/writer contracts, supported formats, key and
  dtype semantics, and factory behavior.

``hyperion.utils``
  Dataset manifests, ``HyperDataset``, Kaldi-style metadata tables,
  enrollment mappings, trial indexes/keys/scores, and supporting utilities.
  These types form the package's core data model.

``hyperion.metrics`` and ``hyperion.np.metrics``
  High-level evaluators and the underlying verification/scoring metrics.

``hyperion.data_prep``
  The ``DataPrep`` extension point and maintained dataset preparation
  implementations.

``hyperion.np``
  NumPy models, PDFs and PLDA, classifiers, transforms, calibration, score
  normalization, preprocessing, clustering, diarization, augmentation, and
  feature utilities.

``hyperion.torch``
  The core PyTorch stack: primitive layers, layer blocks, neural
  architectures, models, datasets/samplers, trainers, losses, optimizers,
  schedulers, loggers, metrics, transforms, and shared utilities. Curated
  reference pages will prioritize documented package exports and model/trainer
  extension points over every implementation module.

``hyperion.torch.tpm``
  Stable integrations with external model and evaluation packages, including
  Hugging Face, DNSMOS, UTMOS, and VoxProfile. Their pages must identify
  optional dependencies and external model requirements.

``hyperion.torch.adv_attacks`` and ``hyperion.torch.adv_defenses``
  Stable adversarial attack and defense interfaces, factories, and supported
  evaluation workflows.

``hyperion.text_norm``
  Text-normalization base classes and English normalizers.

``hyperion.bin``
  All command modules that produce installed ``hyperion-*`` entry points.
  The CLI reference will classify commands by task and document their parser,
  inputs, outputs, and configuration schema.

Experimental package areas
--------------------------

``hyperion.torch.models.dac``
  Neural codec models and streaming codec behavior.

``hyperion.torch.models.freevc`` and VI anonymizer workflows
  VITS-based anonymization and voice-conversion interfaces.

``hyperion.torch.models.transducer`` and ``hyperion.torch.models.wav2transducer``
  Transducer architectures and waveform-to-transducer wrappers.

``hyperion.torch.models.qvectors``
  Q-vector models, wrappers, trainers, and inference workflows.

The associated commands remain part of ``hyperion.bin`` but will be labelled
experimental in the CLI reference.

Out of scope
------------

* ``egs/`` recipes and their shell/tooling dependencies.
* ``hyperion/bin_deprec/`` and ``hyperion/bin_deprec2/``.
* ``hyperion.helpers``: compatibility-oriented composition helpers that are
  currently internal rather than a supported extension surface.
* Private names and incidental implementation modules that are neither
  exported nor required as documented extension points.

Inventory maintenance
---------------------

When adding a package area, command, or top-level extension point, update this
page and :doc:`documentation-policy` in the same pull request if its support
level or scope changes.

See also
--------

* :doc:`documentation-policy`
* :doc:`architecture`
* :doc:`experimental-components`
* :doc:`torch-integrations-and-robustness`
