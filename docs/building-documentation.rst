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

Check public namespace coverage
-------------------------------

``docs/namespace_inventory.json`` is the source of truth for the intended
stable, experimental, and explicitly excluded package namespaces. Every record
has a landing/reference page and an owner page for its documentation contract.
Validate it with:

.. code-block:: bash

   python docs/check_namespace_coverage.py

Add, remove, reclassify, or move a maintained package namespace only with its
inventory record and documented owner page updated in the same pull request.

Check tutorial quality coverage
-------------------------------

``docs/tutorial_inventory.json`` records every maintained tutorial's support
level, prerequisites, expected outputs, and validation path. The checker also
rejects links to ``egs/`` from core-package tutorials:

.. code-block:: bash

   python docs/check_tutorial_coverage.py

Check release notes
-------------------

``docs/release-notes.rst`` has an ``Unreleased`` section for stable public
API, CLI, artifact/configuration compatibility, and deprecation entries. Its
checker requires all categories and replacement/migration links for every
deprecation entry:

.. code-block:: bash

   python docs/check_release_notes.py

Continuous-integration quality gates
------------------------------------

``.github/workflows/docs.yml`` exposes separate ``html``, ``doctest``, and
``linkcheck`` jobs. The HTML and doctest jobs are offline-safe. The linkcheck
job runs in GitHub Actions, where network access is available, and validates
external links and intersphinx inventories. Configure these three workflow
checks as required branch-protection checks for the default branch.

Check CLI coverage and generated index drift
--------------------------------------------

Every maintained ``hyperion/bin/*.py`` module must be classified in
``docs/cli_inventory.json`` as stable, experimental, or explicitly excluded.
The checker also verifies installed command names, matching ``pyproject.toml``
entry points, and assigned family guides:

.. code-block:: bash

   python docs/check_cli_coverage.py
   python docs/render_cli_index.py --check
   python docs/check_cli_quality.py

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
   python docs/check_namespace_coverage.py
   python docs/check_tutorial_coverage.py
   python docs/check_release_notes.py
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

The CI quality gate also verifies that every maintained command is classified,
listed in ``docs/generated/cli-index.rst``, and represented by a top-level
section in ``docs/generated/cli-reference.rst``. Therefore, changing a
maintained parser requires regenerating and committing the help snapshot, even
when its task-family classification stays the same.

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

Run the spelling check for authored RST documentation with the reviewed project
dictionary:

.. code-block:: bash

   ./docs/build.sh spelling

``docs/spelling_wordlist.txt`` contains established toolkit, speech-domain,
and dependency terms such as ``x-vector``, ``PLDA``, ``jsonargparse``,
``Kaldi``, ``VoxProfile``, and ``TPM``. Add a term only after confirming that
it is a deliberate project spelling, not a typo. Generated API documentation,
Python docstrings, and CLI references are intentionally excluded.

External-link retry policy
--------------------------

The strict HTML job validates internal document references and fails
immediately; an unavailable external service never excuses a broken internal
reference. The separate network-enabled linkcheck job retries external-link
and intersphinx failures three times, with a 30-second delay. If all attempts
fail, CI remains failed and its logs identify the external target. Retry the
workflow after a transient provider outage; add an ignore rule only for a
documented, deliberate permanent exception.

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
