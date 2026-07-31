Tutorial Quality Inventory
==========================

``docs/tutorial_inventory.json`` classifies every maintained guided entry
point: getting started, the quickstart, how-to guides, and the speech
augmentation tutorial. Each record identifies its support level, documented
prerequisites, expected outputs, and one validation path.

Validation paths are deliberately limited to:

* a hermetic fixture-scale test;
* a representative CLI smoke test; or
* an explicitly documented hardware, model-asset, optional-dependency, or
  external-data prerequisite.

Core-package tutorials must not link to or depend on ``egs/``. Validate the
inventory before a pull request with:

.. code-block:: bash

   python docs/check_tutorial_coverage.py

When adding a tutorial, add its inventory record and its test or documented
prerequisite in the same pull request. Prefer an executable fixture-scale test
when the workflow can run without a personal corpus, a network download, or
specialized hardware.

See also
--------

* :doc:`contributor-validation`
* :doc:`building-documentation`
