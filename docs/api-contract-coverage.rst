API Contract Coverage
=====================

This inventory defines the high-value public API contract for Phase 4. A listed
concept must have a purpose, input/output constraints, side effects or
exceptions, relevant serialization/configuration behavior, and a focused usage
path in the linked documentation. It is intentionally not an inventory of all
implementation modules.

The machine-readable source of truth is ``docs/api_inventory.json``. Run
``python docs/check_api_coverage.py`` to verify that every classified concept
still has an assigned existing page that mentions it; documentation CI enforces
the same check.

Stable contracts
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Namespace
     - High-value public concepts
     - Contract documentation
   * - ``hyperion.io``
     - Feature reader/writer factories; ``DataReader``/``DataWriter``;
       Ark/HDF5 writers; sequential/random audio readers; VAD factories.
     - :doc:`foundation-api-contracts`, :doc:`io`, and
       :doc:`how-to/prepare-data-and-vad`.
   * - ``hyperion.utils``
     - ``InfoTable`` and manifest subclasses; ``HyperDataset``; dense and sparse
       trial tables; enrollment/class metadata; Kaldi compatibility helpers.
     - :doc:`foundation-api-contracts`, :doc:`data-model`, :doc:`info_tables`,
       :doc:`hyper_dataset`, and :doc:`trials`.
   * - ``hyperion.metrics``
     - EER/DCF functions; verification, adversarial, anonymization,
       speech-quality, and VoxProfile evaluators.
     - :doc:`statistical-api-contracts`, :doc:`metrics`, and
       :doc:`how-to/extract-score-xvectors`.
   * - ``hyperion.data_prep``
     - ``DataPrep`` registration, parser integration, corpus-preparation
       output contract, and manifest validation.
     - :doc:`statistical-api-contracts`, :doc:`data_prep`, and
       :doc:`how-to/prepare-data-and-vad`.
   * - ``hyperion.np``
     - ``HyperNPModel`` serialization; transforms; PLDA factory/scoring;
       calibration; adaptive S-Norm; array metrics.
     - :doc:`statistical-api-contracts`, :doc:`numpy`,
       :doc:`numpy-extension-points`, and :doc:`how-to/save-load-models-and-backends`.
   * - ``hyperion.torch``
     - ``HyperTorchModel`` serialization/train modes; waveform x-vectors;
       ``NetArch`` forward/mask contract; datasets/samplers; trainers;
       optimizers and schedulers; stable layers/blocks/narchs.
     - :doc:`torch-api-contracts`, :doc:`torch-api`,
       :doc:`torch-extension-points`, :doc:`torch-layers-and-architectures`,
       and :doc:`torch-training-support`.
   * - ``hyperion.text_norm``
     - Basic/English text normalizers, English number normalizers, and spelling
       normalization behavior.
     - :doc:`statistical-api-contracts` and :doc:`text_norm`.

Experimental or intentionally excluded surfaces
------------------------------------------------

``hyperion.torch.models.dac``, VITS/freevc anonymization, transducers, and
Q-vector models are documented as experimental in :doc:`experimental-components`.
Their APIs may be described for evaluation, but they are not a stable extension
contract.

``hyperion.helpers`` is not currently a supported public namespace. It is
therefore deliberately excluded from this inventory rather than receiving an
implicit API guarantee. If a helper becomes supported, add it to
:doc:`public-surface`, this table, and a contract page in the same change.

Review rule
-----------

When a change adds or materially changes a listed public concept, update its
linked contract page in the same pull request. A generated signature alone is
not sufficient: the documentation must state data layout, compatibility, and
observable behavior needed by callers.
