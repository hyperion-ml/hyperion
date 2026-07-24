Visualization and Utility Commands
==================================

These commands operate on existing artifacts and should not change their key
semantics.

``hyperion-merge-scores`` and ``hyperion-merge-trials`` combine compatible
score/trial shards. Input model/segment axes and masks must be aligned before
merging. Write a new output rather than overwriting source shards.

``hyperion-plot-embedding-tsne`` and
``hyperion-plot-embedding-tsne-per-class`` create visualization artifacts from
embeddings and optional labels. They require ``matplotlib`` and
``scikit-learn``. t-SNE plots are exploratory diagnostics, not a verification
metric or a substitute for trial-based evaluation.

See also
--------

* :doc:`../foundation-api-contracts`
* :doc:`../metrics`
