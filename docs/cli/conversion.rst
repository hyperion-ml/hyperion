Audio, Feature, and VAD Conversion Commands
============================================

This family converts existing artifacts without changing their semantic ids.
Use CSV indexes for new outputs.

PyTorch checkpoint migration
----------------------------

``hyperion-to-safetensors`` converts a trusted legacy Hyperion model ``.pth``
checkpoint to the modern ``config.json`` plus ``model.safetensors`` layout. It
defaults to a ``model/`` subdirectory under ``--out-model-dir``; use
``--model-name`` for names such as ``dac_model`` or ``discrim_model``. Add
``--get-trainer-state`` when the converted checkpoint must resume a modern
trainer rather than only load a model for inference.

Legacy conversion loads the input with PyTorch ``weights_only=False`` to read
its pickled metadata. Only convert files from trusted sources. See
:doc:`../how-to/manage-torch-checkpoints` for directory layouts, multi-model
conversion, and examples.

VAD conversion
--------------

``hyperion-convert-vad-format`` converts VAD representations. It accepts a
VAD input and output specifier; paired Ark/CSV output is supported:

.. code-block:: bash

   hyperion-convert-vad-format \
   --in-vad-file data/eval/vad.csv \
   --out-vad-file ark,csv:data/eval/vad.ark,data/eval/vad.csv

When reading that binary output back, use the CSV index rather than a bare Ark
path. The index supplies the keys and random-access offsets:

.. code-block:: bash

   hyperion-convert-vad-format bin_to_time_marks \
     --in-vad-file csv:data/eval/vad.csv \
     --out-vad-file data/eval/vad_time_marks.csv \
     --output-dir data/eval/vad_time_marks

When binary VAD is involved, preserve frame length, frame shift, and
``snip_edges`` settings. A frame/time mismatch changes speech boundaries.

Audio and feature utilities
---------------------------

``hyperion-preprocess-audio-files``, ``hyperion-audio-to-duration``,
``hyperion-compute-mfcc-feats``, ``hyperion-compute-energy-vad``,
``hyperion-copy-feats``, ``hyperion-apply-mvn-select-frames``,
``hyperion-pack-wav-rirs``, and ``hyperion-make-babble-noise-audio-files``
operate on audio or feature artifacts. Audio-reading commands require an
available audio backend. Record the output sample rate, feature configuration,
and metadata columns alongside generated CSV indexes.

See also
--------

* :doc:`../how-to/prepare-data-and-vad`
* :doc:`../io`
* :doc:`../foundation-api-contracts`
