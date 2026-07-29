Command-Line Interface
======================

Entry point model
-----------------

Hyperion CLI commands are generated from scripts in ``hyperion.bin``
(``hyperion/bin`` on disk).

Generation flow:

1. ``proto_pyproject.toml`` defines static package metadata template.
2. ``generate_pyproject.py`` scans ``hyperion/bin/*.py``.
3. Each script is mapped to ``hyperion-<script-name>`` in ``pyproject.toml``.
4. Dependencies are loaded from ``requirements.txt``.

The machine-readable classification source is ``docs/cli_inventory.json``. It
contains one record for every maintained executable module in ``hyperion/bin``:
the installed command name, task family, stable/experimental status, planned
guide page, and optional runtime requirements. Later CLI reference pages and CI
checks are generated from or validated against this inventory.

Regenerate entry points with:

.. code-block:: bash

   python generate_pyproject.py

Naming convention
-----------------

A script file like ``hyperion/bin/train_qvector.py`` becomes command:

.. code-block:: bash

   hyperion-train-qvector

Task-oriented command index
----------------------------

The following complete index is generated from ``docs/cli_inventory.json``.
Stable commands are supported public interfaces. Experimental commands are
visible for evaluation, but their configuration and checkpoint compatibility
are not guaranteed across minor releases. Family guides with full workflows
are linked below.

.. include:: generated/cli-index.rst

Family guides
-------------

.. toctree::
   :maxdepth: 1

   cli/training
   cli/fine-tuning
   cli/extraction-inference
   cli/backend-scoring-evaluation
   cli/data-preparation
   cli/conversion
   cli/visualization-utilities
   cli/experimental

Generated command and option reference
--------------------------------------

The complete parser-derived option reference is generated from command
``--help`` output. It is included separately because it is intentionally an
exact parser snapshot rather than a task guide:

.. toctree::
   :maxdepth: 1

   generated/cli-reference

Scope notes
-----------

* ``hyperion/bin_deprec`` and ``hyperion/bin_deprec2`` are deprecated and not
  part of the documented CLI surface.
* Experimental transducer decoders are listed in
  :doc:`experimental-components`.

Optional dependencies and TPM commands
--------------------------------------

The generated index labels **conditional runtime requirements**: dependencies,
extras, model assets, or network retrieval needed for that command, but not for
every Hyperion workflow. This does not promise that an asset is installed,
licensed, available offline, or compatible with an arbitrary device.

The policy for package extras, local assets, first-run downloads, offline
execution, and stable TPM wrappers is defined in :doc:`optional-dependencies`.
For reproducible work, provide local assets and pin model revisions rather than
depending on a mutable remote default or a personal cache.

Discover commands
-----------------

After installation:

.. code-block:: bash

   python -m pip show hyperion-ml
   hyperion-train-qvector --help
   hyperion-eval-verification-metrics --help
