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

* V2 Transformer encoder and QFormer architectures support optional QK/value
  RMS normalization, pre/post normalization, configurable head widths, K=V
  projections, GELU with tanh approximation, and dense-plus-routed G4MoE blocks.
  Encoder local/global attention has independent RoPE settings, KV sharing,
  and causal convolution streaming. Padding and output lengths now account
  for exact convolutions and endpoint resampling. The final superblock is
  always included in multi-layer aggregation, and invalid tensor-parallel
  head partitions are rejected. See :doc:`torch-layers-and-architectures` and
  :doc:`torch-api-contracts`.
* DDP trainers consult the top-level model's unused-parameter requirement;
  composed task models must delegate to conditional encoders. See
  :doc:`torch-training-support`.

CLI commands
~~~~~~~~~~~~

No maintained CLI commands have been added, removed, or renamed for the next
release.

Artifact and configuration compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* V2 encoder configurations use local/global window and RoPE arguments in
  place of the former shared names. QFormer has separate self/cross K=V
  options. Norm precision uses ``norm_eps``; new projection/norm/MoE choices
  can change checkpoint keys and shapes. Dynamic-reference RoPE now retains
  frequency scaling in training. See :doc:`torch-api-contracts` for argument
  migration and :doc:`torch-layers-and-architectures` for defaults.
* ``flash_attention_version`` selects native Torch or HF versions; native
  selection is process-wide and HF/newer-kernel compatibility remains
  conditional. Configuration/checkpoint loading does not restore external
  inference cache state. See :doc:`torch-layers-and-architectures`.

Deprecations
~~~~~~~~~~~~

No stable-interface deprecations have been recorded for the next release.

Released versions
-----------------

Add a release heading below this section when publishing a version. Preserve
older entries so users can trace API, CLI, artifact, configuration, and
deprecation history.
