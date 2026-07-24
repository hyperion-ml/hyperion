Train a Pretrained Wav2Vec2 X-Vector
====================================

Use this guide when a Hugging Face speech encoder should provide the acoustic
representation for an x-vector speaker classifier. This is distinct from the
native frontend workflow in :doc:`train-waveform-xvector`: it introduces
pretrained weights, a Hugging Face cache, feature fusion, and fine-tuning
policy choices.

Choose an architecture
----------------------

Start with the ResNet1d head unless an experiment specifically requires a
Conformer head:

.. code-block:: bash

   hyperion-train-wav2vec2xvector hf_wav2vec2resnet1d --help

The command also supports WavLM, HuBERT, Whisper, and Wav2Vec2-BERT wrappers.
Choose the wrapper that matches the intended pretrained checkpoint; do not
assume weights from one encoder family are interchangeable with another.

Configure the pretrained encoder
--------------------------------

The model configuration has three principal sections:

.. code-block:: yaml

   model:
     hf_feats:
       pretrained_model_path: /models/wav2vec2-base
       cache_dir: /shared/hyperion-hf-cache
     feat_fuser:
       # Layer-fusion configuration.
     xvector:
       # ResNet1d x-vector-head configuration.

``pretrained_model_path`` accepts a local checkpoint directory or a Hugging
Face Hub identifier. A local directory is the preferred choice for
reproducible or air-gapped training. Set ``cache_dir`` to shared, persistent
storage rather than a node-local temporary directory.

For a Hub identifier, run a small validation job with network access first so
the model and processor are cached. Offline workers must be able to resolve the
same cached files or local model directory before distributed training starts.

Fine-tuning policy
------------------

The trainer's ``train_mode`` selects which portions are trainable. The
available modes include:

* ``ft-xvector``: train the x-vector head while freezing the Hugging Face
  encoder and feature fuser.
* ``hf-feats-frozen``: train the x-vector path while retaining the full
  pretrained encoder as frozen features.
* ``hf-feat-extractor-frozen``: freeze the waveform feature extractor while
  allowing higher encoder layers to adapt.
* ``full``: fine-tune all components.
* ``hf-lora`` and bias/LoRA variants: adapt the Hugging Face encoder through
  LoRA-oriented policies.

Begin with ``ft-xvector`` or ``hf-feats-frozen`` when labeled speaker data is
limited. Move to partial or full fine-tuning only after establishing a stable
baseline. Full fine-tuning needs considerably more GPU memory and a smaller
learning rate than head-only training.

Train and extract
-----------------

.. code-block:: bash

   hyperion-train-wav2vec2xvector hf_wav2vec2resnet1d \
     --cfg configs/hf-wav2vec2-resnet1d.yaml

Use the matching extractor after training:

.. code-block:: bash

   hyperion-extract-wav2vec2xvectors \
     --dataset-path data/eval/dataset.yaml \
     --model-path exp/hf-wav2vec2-resnet1d/<checkpoint> \
     --xvector-path ark,csv:exp/eval/xvectors.ark,exp/eval/xvectors.csv \
     --use-gpu

The extractor resamples to the model's sample frequency. Do not rely on this
as a substitute for validating the training corpus sample rate: the encoder,
dataset, and augmentation settings should be designed around one rate.

Memory and performance
----------------------

Pretrained encoders retain hidden states and are sensitive to waveform length.
Use the following order when memory is constrained:

1. Reduce sampler maximum batch duration.
2. Enable AMP and select a supported dtype.
3. Use a frozen-head training mode.
4. Reduce the number of fused encoder layers or use a smaller backbone.
5. Use distributed training as described in
   :doc:`run-resumable-distributed-training`.

See also
--------

* :doc:`train-waveform-xvector`
* :doc:`extract-score-xvectors`
* :doc:`run-resumable-distributed-training`
