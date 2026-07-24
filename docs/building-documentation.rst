Building the Documentation
==========================

The documentation is built from the ``docs/`` directory with Sphinx. The
canonical build command is the repository script, which works from any current
directory and writes HTML to ``docs/_build/html``.

Install documentation dependencies
----------------------------------

Use the Python environment in which you work on Hyperion, or create a separate
virtual environment for documentation work:

.. code-block:: bash

   python3 -m pip install -r docs/requirements.txt

Build HTML
----------

From the repository root:

.. code-block:: bash

   ./docs/build.sh

Check public API coverage
-------------------------

The curated public API inventory is stored in ``docs/api_inventory.json``.
Validate that every stable, experimental, or explicitly excluded concept is
assigned to documentation that mentions it:

.. code-block:: bash

   python docs/check_api_coverage.py

The documentation CI workflow runs this check before the strict HTML build.

Check CLI coverage and generated index drift
--------------------------------------------

Every maintained ``hyperion/bin/*.py`` module must be classified in
``docs/cli_inventory.json`` as stable, experimental, or explicitly excluded.
The checker also verifies installed command names, matching ``pyproject.toml``
entry points, and assigned family guides:

.. code-block:: bash

   python docs/check_cli_coverage.py
   python docs/render_cli_index.py --check

The full-runtime CI job additionally checks the generated option reference.
It uses ``--allow-unavailable`` only so that existing parser-import diagnostics
remain a reproducible part of the reference; it does not hide them from the
generated page.

Run documentation CI locally before a pull request
--------------------------------------------------

From the repository root, run the lightweight CI job with:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python docs/check_api_coverage.py
   python docs/check_cli_coverage.py
   python docs/render_cli_index.py --check
   HYPERION_PYTHON=python docs/build.sh html
   HYPERION_PYTHON=python docs/build.sh linkcheck

Then run the slower full-runtime CLI reference check with its pinned
environment:

.. code-block:: bash

   python -m pip install -r docs/requirements-cli-reference.txt
   python docs/generate_cli_reference.py \
     --python python \
     --check \
     --allow-unavailable \
     --timeout 90 \
     --jobs 4

The same runtime also runs representative end-to-end commands for table
operations, VAD conversion, score merging, and verification metrics:

.. code-block:: bash

   python -m pytest tests/docs/test_cli_end_to_end.py

If either generated-file check reports drift, regenerate the corresponding
file as described below, review the changes, and rerun the check.

Regenerate the CLI option reference
-----------------------------------

``docs/generated/cli-reference.rst`` is a checked-in snapshot of the actual
``jsonargparse`` help text. It prevents hand-maintained option tables from
drifting away from the installed commands. Regenerate it after changing a
maintained CLI parser or its registered classes:

.. code-block:: bash

   python -m pip install -r docs/requirements-cli-reference.txt
   python docs/generate_cli_reference.py --python python

The generator captures every command in ``docs/cli_inventory.json`` and the
help of every listed jsonargparse subcommand. It exits unsuccessfully if an
entry point cannot load, which normally means an optional dependency is absent
or the parser itself is broken. For dependency-diagnostic work only, use
``--allow-unavailable``; do not publish that diagnostic output as a complete
reference. ``--script MODULE_STEM`` limits generation to one inventory entry,
and ``--jobs`` controls the number of concurrent help processes.

``docs/requirements-cli-reference.txt`` is deliberately pinned to the runtime
used by the checked-in snapshot and its CI drift check. Update it and regenerate
the reference together when changing the supported CLI runtime.

The build treats warnings as errors. This is deliberate: unresolved links,
invalid cross-references, and malformed reStructuredText must be fixed before
the documentation is published.

The script chooses ``python`` when available, otherwise ``python3``. Hyperion
requires Python 3.10 or newer. To select a specific compatible environment,
set ``HYPERION_PYTHON`` to its Python executable:

.. code-block:: bash

   HYPERION_PYTHON=/path/to/python ./docs/build.sh

The generated site is located at ``docs/_build/html/index.html``.

Offline and online builds
-------------------------

HTML and doctest builds are offline-safe by default: they do not fetch external
intersphinx inventories. This keeps the strict local build reliable on an
air-gapped machine or an intermittent connection. External Python references
will still render as code text when an inventory is unavailable.

For an online documentation or CI build that should resolve those references,
set ``HYPERION_DOCS_ONLINE=1``:

.. code-block:: bash

   HYPERION_DOCS_ONLINE=1 ./docs/build.sh html

Additional checks
-----------------

Run the external-link checker:

.. code-block:: bash

   ./docs/build.sh linkcheck

``linkcheck`` enables ``HYPERION_DOCS_ONLINE`` automatically and therefore
requires network access.

Run executable documentation examples after they are added:

.. code-block:: bash

   ./docs/build.sh doctest

The doctest target runs only examples deliberately marked with Sphinx
``testcode``/``testoutput`` directives. Ordinary ``>>>`` sessions in API
docstrings are explanatory examples and may require corpus fixtures or optional
dependencies, so they are rendered but not executed. Make a new example
executable only when it is hermetic and its required fixture-scale inputs are
part of the repository.

To remove generated documentation files:

.. code-block:: bash

   ./docs/build.sh clean

Contributing documentation
--------------------------

Use the support boundaries in :doc:`documentation-policy` and update the
relevant guide or reference page when a public interface changes. Keep examples
package-focused and fixture-scale; ``egs/`` recipes are outside this
documentation scope.
