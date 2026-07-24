Deprecation and Compatibility Policy
====================================

This policy governs maintained public interfaces in ``hyperion``: documented
Python imports, installed commands from ``hyperion.bin``, configuration files,
serialized model artifacts, and documented data formats. It defines how an
interface changes without forcing users to reverse-engineer a replacement.

Support boundary
----------------

Stable interfaces preserve ordinary use across compatible releases. In
practice, this includes public names, documented call and parser signatures,
configuration keys and defaults, checkpoint/configuration fields, CSV columns
and ID semantics, and documented output formats. A stable change must either
be backward compatible or follow the deprecation process below.

Experimental codec/DAC, VITS anonymization/freevc, transducer, and q-vector
interfaces do not promise cross-minor-release API, configuration, or checkpoint
compatibility. They must still state their support level and prerequisites. Do
not silently make an experimental interface look stable merely because an old
artifact happens to load.

``egs/``, ``hyperion/bin_deprec/``, ``hyperion/bin_deprec2/``, private names,
and implementation-only modules are outside this policy. A migration document
may mention them only to identify a maintained replacement.

What needs a compatibility decision
-----------------------------------

.. list-table:: Public change categories
   :widths: 29 35 36
   :header-rows: 1

   * - Interface
     - Compatible change
     - Deprecation or migration required
   * - Python API
     - Add an optional parameter with a behavior-preserving default; add a new
       method or class.
     - Rename/remove a public import, argument, method, return field, error
       convention, or change a documented default/meaning.
   * - CLI and configuration
     - Add an optional flag or nested configuration key with a stable default.
     - Rename/remove a command, subcommand, option, config key, accepted value,
       default, output path, or generated artifact meaning.
   * - NumPy/PyTorch artifact
     - Add optional configuration data that old loaders safely ignore, while
       retaining existing class and parameter names.
     - Change ``class_name``, state-dict/parameter key, tensor shape, class
       inventory, serialized configuration meaning, or loader behavior.
   * - CSV and other data artifacts
     - Add an optional column without changing existing columns or IDs.
     - Rename/remove a column, alter ID/time/score semantics, change dtype or
       ordering guarantees, or stop reading an established format.

Changing a default is a breaking behavior change when users may reasonably
have relied on the old documented behavior. Adding a more restrictive
validation rule is also breaking if files or configurations that were valid
before now fail.

Deprecation lifecycle
---------------------

Use this lifecycle for a stable interface that needs replacement:

#. **Choose a maintained replacement.** Do not deprecate an interface before a
   usable supported alternative exists, except for a security or correctness
   emergency. State the replacement, behavior difference, and artifact
   migration in the relevant guide.
#. **Keep a compatibility path.** Retain the old Python name, option, config
   key, serialized field, or reader path and translate it to the replacement
   internally. The compatibility path must preserve old behavior, not merely
   accept the old spelling and silently change its result.
#. **Make the deprecation visible.** Python APIs emit a targeted
   ``DeprecationWarning`` at the public call boundary. CLI help marks the old
   option/subcommand deprecated and runtime invocation emits a clear warning to
   stderr or logging. Documentation identifies the old interface, replacement,
   migration example, and intended removal release when known.
#. **Test both paths.** Add a regression test for the legacy spelling/artifact
   and a test for the replacement. For a loader migration, include a fixture
   from the old representation or construct an equivalent old-format artifact.
#. **Remove deliberately.** Remove only after the published deprecation window
   has elapsed and the release notes repeat the migration. If no concrete
   release can be named, leave the compatibility path in place rather than
   making an unannounced removal.

For normal stable changes, the deprecation window is at least one published
release after the first release that documents the replacement. Security,
data-loss, or demonstrably incorrect behavior may require a faster removal;
the release notes and migration documentation must explain why and give the
safest available alternative.

Implementation patterns
-----------------------

Python APIs
~~~~~~~~~~~

Keep a deprecated alias or keyword at the public boundary, translate it once,
and warn once. Reject ambiguous use of an old and new argument together with a
clear error. Do not spread compatibility branches through internal code. Put
the translation next to the public constructor/function and retain a test that
can be removed with the alias.

CLI and ``jsonargparse`` configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep a deprecated command option or subcommand parser long enough for existing
shell scripts and YAML/JSON configuration to work. Map an old option to the
same destination as the replacement or translate the parsed namespace before
calling the implementation. The generated ``--help`` reference is part of the
compatibility evidence: regenerate it and ensure the deprecated status and new
path are readable.

When a config key moves into a nested group, accept and migrate the old key at
the parser/normalization boundary. Avoid changing a default in the same
release as a rename; users need one predictable transition at a time. Update
the command-family guide and add a config-file test for both forms.

Models and checkpoints
~~~~~~~~~~~~~~~~~~~~~~

``class_name`` and model configuration are persisted public data. Keep old
class names resolvable, or map them explicitly in the loader. Perform narrow,
versioned migrations near ``load()``/``auto_load()`` or the maintained model
loader; do not make every model constructor interpret arbitrary historical
configuration.

Retain or map old parameter/buffer keys and validate shape changes before
loading. Never silently reshape or drop learned state that changes model
meaning. If a checkpoint cannot be migrated safely, fail with an error that
names the incompatible field and points to the required conversion or retained
runtime. See :doc:`model-extension-contracts` for the base serialization
contract.

Data formats
~~~~~~~~~~~~

Readers should continue to accept stable documented artifacts while writers
produce the current canonical representation. Preserve key/ID alignment,
dtypes, time units, score semantics, and required CSV columns. A format change
needs a reader compatibility test and a round-trip or downstream-consumer test.

New public manifest examples are CSV-based. Do not introduce new ``.scp``
workflows; where a binary Ark payload is needed, use and document an
``ark,csv:`` output with its CSV index.

Removal checklist
-----------------

Before removing a deprecated stable interface, confirm all of the following:

* a documented maintained replacement has been available for at least one
  published release, unless an emergency exception was documented;
* the removal release and migration instructions appeared in documentation and
  release notes;
* public guides, API contracts, CLI inventory, and generated command help no
  longer present the removed interface as current;
* migration or conversion tooling exists when an artifact cannot be read by
  the replacement directly;
* tests cover the replacement and any remaining artifact-compatibility path;
  and
* affected optional dependencies, TPM assets, and experimental status are
  described accurately.

Documenting a deprecation
-------------------------

Each deprecation notice must answer four questions near the affected API or
command: what is deprecated, what replaces it, how a user migrates code/config
or files, and when removal is intended. For stable interfaces, add a concise
migration entry to the relevant task guide and release notes. For CLI changes,
also update ``docs/cli_inventory.json`` and regenerate the CLI index/reference
when command classification or parser help changes.

The pull request should identify compatibility impact explicitly and list the
old/new tests, generated documentation updates, and any exception to the
normal deprecation window. See :doc:`contributor-validation` for the complete
validation checklist.

See also
--------

* :doc:`documentation-policy`
* :doc:`contributor-extension-guide`
* :doc:`contributor-validation`
* :doc:`model-extension-contracts`
* :doc:`data-preparation-and-cli-extensions`
* :doc:`building-documentation`
