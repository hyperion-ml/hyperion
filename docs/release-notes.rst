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

No stable public API changes have been recorded for the next release.

CLI commands
~~~~~~~~~~~~

No maintained CLI commands have been added, removed, or renamed for the next
release.

Artifact and configuration compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No artifact, checkpoint, serialized configuration, or stable data-format
compatibility changes have been recorded for the next release.

Deprecations
~~~~~~~~~~~~

No stable-interface deprecations have been recorded for the next release.

Released versions
-----------------

Add a release heading below this section when publishing a version. Preserve
older entries so users can trace API, CLI, artifact, configuration, and
deprecation history.
