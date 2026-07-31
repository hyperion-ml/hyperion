Documentation Policy
====================

Scope
-----

The maintained documentation describes the ``hyperion`` Python package and
the command-line interfaces generated from ``hyperion/bin``. It does not
describe experiment recipes in ``egs/`` or legacy scripts in
``hyperion/bin_deprec`` and ``hyperion/bin_deprec2``.

The documentation distinguishes public APIs from internal implementation
details. A public API is importable or executable as a supported Hyperion
interface and is intentionally documented. An implementation detail may
appear in generated API pages only when it is needed to explain a public
extension point.

Support levels
--------------

Stable
~~~~~~

Stable components are supported public interfaces. Changes to their ordinary
usage, configuration, serialization, or output formats require migration
guidance and release-note coverage.

* ``hyperion.io``
* ``hyperion.utils``
* ``hyperion.metrics``
* ``hyperion.data_prep``
* ``hyperion.np``
* core ``hyperion.torch`` layers, data, models, trainers, and training support
  interfaces
* ``hyperion.torch.tpm`` third-party model wrappers
* ``hyperion.torch.adv_attacks`` and ``hyperion.torch.adv_defenses``
* ``hyperion.text_norm``
* maintained commands generated from ``hyperion/bin``

Experimental
~~~~~~~~~~~~

Experimental components are documented and usable, but their APIs,
configuration schemas, checkpoints, and behavior may change in a minor
release without the compatibility guarantees of stable components. Their
documentation must carry an experimental notice and state the relevant
optional dependencies or hardware requirements.

* neural codec models and commands, including ``hyperion.torch.models.dac``
* VITS-based voice anonymization and voice-conversion models and commands,
  including ``hyperion.torch.models.freevc`` and VI anonymizer workflows
* transducer and waveform-to-transducer models and commands, including
  ``hyperion.torch.models.transducer`` and
  ``hyperion.torch.models.wav2transducer``
* q-vector models and commands, including ``hyperion.torch.models.qvectors``

Internal and legacy
~~~~~~~~~~~~~~~~~~~

Private names, implementation-only helpers, and code whose module path is not
part of the documented public surface are internal. Legacy command trees are
not part of the maintained interface. They may be mentioned only in a
migration notice that identifies a supported replacement.

Documentation requirements
--------------------------

Every change to a public interface must update the relevant documentation in
the same pull request.

* New public Python APIs need a reference entry, complete docstrings, and a
  link from a task-oriented guide or concept page when users need context to
  choose or configure them.
* New maintained commands need a command-index entry, a minimal invocation,
  documented inputs and outputs, and configuration-file coverage when they
  use ``jsonargparse``.
* Stable APIs need input/output contracts, including array or tensor shape,
  dtype, filesystem effects, serialization behavior, and meaningful errors.
* Experimental APIs need an experimental notice, prerequisites, and known
  compatibility limits.
* Deprecated APIs need a replacement, a migration path, and the intended
  removal release when known. Follow :doc:`deprecation-and-compatibility` for
  the compatibility window, warnings, and removal conditions.

Quality gates
-------------

The documentation build is a required engineering check. The eventual CI
workflow must build HTML with warnings treated as errors, check links, and run
selected executable examples. Documentation examples must use package-level
workflows and fixture-scale data; they must not require ``egs/`` recipes.

See also
--------

* :doc:`contributor-extension-guide`
* :doc:`contributor-validation`
* :doc:`deprecation-and-compatibility`
* :doc:`public-surface`
* :doc:`architecture`
* :doc:`experimental-components`
