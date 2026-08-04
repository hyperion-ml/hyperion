Release Notes
=============

This page records changes to Hyperion's maintained public surface. Update the
``Unreleased`` section in the same pull request as any stable public API,
maintained CLI, artifact/configuration compatibility, or deprecation change.
Experimental-only changes may be noted when useful, but they do not replace
the required stable-interface entries.

Entry rules
-----------

Write entries for user-visible behavior, not internal refactors. State the
affected import, command, configuration key, artifact, or format and link to
the relevant guide or API contract. New, removed, and renamed maintained CLI
commands must name both old and new commands where applicable.

Every deprecation entry uses this one-line format so its replacement and
migration are unambiguous:

.. code-block:: rst

   * **Deprecated:** ``old-interface``. **Replacement:** :doc:`new-guide`.
     **Migration:** :doc:`migration-guide`. **Removal target:** 0.x.

Both the **Replacement** and **Migration** fields must be documentation links.
See :doc:`deprecation-and-compatibility` for the compatibility window and
implementation requirements.

Unreleased
----------

Stable public API
~~~~~~~~~~~~~~~~~

* ``HyperTorchModel.save`` and ``HyperTorchModel.auto_load`` now support
  model directories containing ``config.json`` and ``model.safetensors`` in
  addition to the legacy ``.pth`` model format. See
  :doc:`how-to/save-load-models-and-backends`.

CLI commands
~~~~~~~~~~~~

* Added ``hyperion-to-safetensors`` to convert trusted legacy Hyperion model
  checkpoints to model directories, optionally including trainer-resume state.
  See :doc:`cli/conversion`.

Artifact and configuration compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Modern ``TorchTrainerBase`` subclasses now publish atomic checkpoint
  directories with JSON model configuration, safetensors model weights, and
  separate PyTorch optimizer and scheduler state. Existing legacy ``.pth``
  model checkpoints remain loadable and can be migrated with
  ``hyperion-to-safetensors``. See :doc:`how-to/manage-torch-checkpoints`.

Deprecations
~~~~~~~~~~~~

No stable-interface deprecations have been recorded for the next release.

Released versions
-----------------

Add a release heading below this section when publishing a version. Preserve
older entries so users can trace API, CLI, artifact, configuration, and
deprecation history.
