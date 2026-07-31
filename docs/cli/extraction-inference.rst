Extraction and Inference Commands
=================================

These commands turn saved models and CSV manifests into embeddings, logits, or
decoded outputs. All stable embedding extractors preserve segment ids in their
output CSV index; those ids must align with enrollment and trial tables.

Embedding extraction
--------------------

For a native waveform checkpoint:

.. code-block:: bash

   hyperion-extract-wav2xvectors \
     --dataset-path data/eval/dataset.yaml \
     --model-path exp/wav2xvector/<checkpoint> \
     --xvector-path ark,csv:exp/eval/xvectors.ark,exp/eval/xvectors.csv \
     --use-gpu

For a Hugging Face waveform model, use
``hyperion-extract-wav2vec2xvectors`` with the matching checkpoint. Feature and
legacy waveform paths use ``hyperion-extract-xvectors-from-wav``. Sliding-window variants additionally
produce window-level embeddings and timestamps.

Hugging Face extraction has conditional ``transformers`` and pretrained-asset
requirements. Pin the model revision and pre-populate a local cache or pass a
local model path before an offline run; see :doc:`../optional-dependencies`.

Inputs and artifacts
--------------------

Inputs are a dataset manifest or compatible recording/segment CSV files plus a
model checkpoint. Outputs are Ark/HDF5 payloads with a CSV index. The index is
the supported interchange artifact; do not create new workflows around SCP.
The model sample frequency, frontend, and embedding dimension must match the
checkpoint configuration.

Logit evaluation
----------------

``hyperion-eval-wav2xvector-logits``,
``hyperion-eval-wav2vec2xvector-logits``, and
``hyperion-eval-xvec-logits-from-wav`` run classifier outputs against labeled
data. They require class ids compatible with the checkpoint classifier head and
write the requested reports/artifacts according to parser options.

Experimental inference
----------------------

Transducer decoders and ``hyperion-infer-qvectors`` are experimental. They may
need additional model assets and have no stable checkpoint/configuration
guarantee; see :doc:`experimental`.

See also
--------

* :doc:`../how-to/extract-score-xvectors`
* :doc:`../foundation-api-contracts`
* :doc:`../torch-api-contracts`
