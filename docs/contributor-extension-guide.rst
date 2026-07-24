Contributor Extension Guide
===========================

This guide is the entry point for contributors extending the maintained
``hyperion`` package. It describes where a change belongs and which existing
contracts it must preserve. It deliberately does not treat ``egs/`` recipes or
``hyperion/bin_deprec*`` as extension templates: those trees are outside the
supported product surface.

Choose the extension point
--------------------------

Start with the smallest maintained component that owns the new behavior.

.. list-table:: Maintained extension points
   :widths: 26 32 42
   :header-rows: 1

   * - You are adding
     - Put it under
     - Start with
   * - A serializable NumPy model, transform, PDF, calibrator, or score method
     - ``hyperion.np``
     - :doc:`model-extension-contracts` and :doc:`numpy-extension-points`
   * - A primitive PyTorch operation
     - ``hyperion.torch.layers``
     - :doc:`torch-extension-workflows`
   * - A reusable composition of layers
     - ``hyperion.torch.layer_blocks``
     - :doc:`torch-extension-workflows`
   * - A reusable neural architecture
     - ``hyperion.torch.narchs``
     - ``NetArch`` and :doc:`torch-extension-workflows`
   * - A task model, model checkpoint, or model-specific inference behavior
     - ``hyperion.torch.models``
     - :doc:`model-extension-contracts` and :doc:`torch-api-contracts`
   * - Batching, sampling, training, logging, or optimization behavior
     - ``hyperion.torch.data`` or ``hyperion.torch.trainers``
     - :doc:`torch-extension-workflows` and :doc:`torch-training-support`
   * - Dataset-specific manifest preparation
     - ``hyperion.data_prep``
     - :doc:`data-preparation-and-cli-extensions`
   * - A maintained command
     - ``hyperion.bin``
     - :doc:`data-preparation-and-cli-extensions`

Architectural boundaries
-------------------------

The PyTorch stack has a deliberate dependency direction:

``layers`` → ``layer_blocks`` → ``narchs`` → ``models`` → ``trainers``.

Keep reusable computation below the model layer. A model owns task-specific
forward/loss/embedding behavior; a trainer owns optimization, checkpointing,
distributed execution, AMP, and logging. New dataset or sampler behavior must
preserve the selected trainer's batch contract. See :doc:`architecture` for the
package layout and :doc:`torch-extension-points` for the current public
contracts.

For NumPy components, keep embedding transforms, PLDA/PDFs, score
normalization, calibration, and evaluation as independently serializable
artifacts. Do not couple a score-side component to a PyTorch checkpoint unless
the public model contract explicitly requires it.

Contributor responsibilities
----------------------------

Before making a public extension, identify these decisions:

* **Public contract:** constructor arguments, types, shapes, outputs, errors,
  and side effects.
* **Configuration and serialization:** whether the component participates in a
  registry, needs ``get_config()``, and must load artifacts produced by earlier
  releases.
* **Configuration-facing integration:** whether a factory or
  ``jsonargparse`` class-argument interface is preferable to custom CLI
  plumbing.
* **Support level:** stable additions require compatibility and migration
  planning; experimental additions must be labelled as such. See
  :doc:`documentation-policy`.
* **Validation:** select targeted tests, update a public CLI inventory entry if
  applicable, regenerate derived CLI docs, and run the checks in
  :doc:`building-documentation`.

The Phase 6 workflow pages expand these conventions with maintained code
patterns. Use the nearest maintained component in the target subsystem as the
implementation pattern, not a similarly named legacy script.

See also
--------

* :doc:`architecture`
* :doc:`documentation-policy`
* :doc:`model-extension-contracts`
* :doc:`torch-extension-workflows`
* :doc:`data-preparation-and-cli-extensions`
* :doc:`contributor-validation`
* :doc:`deprecation-and-compatibility`
* :doc:`torch-extension-points`
* :doc:`numpy-extension-points`
* :doc:`cli`
