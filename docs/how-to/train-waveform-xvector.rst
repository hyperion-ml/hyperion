Train an X-Vector Directly from Waveforms
==========================================

Use this guide to train a speaker-classification x-vector model from audio
files. Hyperion reads waveforms, applies the configured frontend, trains an
x-vector classifier, and stores checkpoints under the trainer experiment
directory. This is the waveform counterpart to the synthetic scoring workflow
in :doc:`../quickstart`.

Choose a model family
---------------------

``hyperion-train-wav2xvector`` is the native waveform path. It computes
acoustic features in the model and supports these subcommands:

* ``resnet``
* ``resnet1d``
* ``conformer``
* ``convnext1d``
* ``convnext2d``
* ``transformer_v2``

For a pretrained Hugging Face backbone followed by an x-vector head, use
``hyperion-train-wav2vec2xvector``. Its subcommands include
``hf_wav2vec2resnet1d`` and ``hf_wav2vec2conformer`` as well as HuBERT,
WavLM, Whisper, and Wav2Vec2-BERT variants. This path needs the optional
Hugging Face dependencies, access to the requested pretrained weights, and
substantially more accelerator memory.

Start with the native ``resnet1d`` workflow unless a pretrained encoder is a
deliberate requirement.

Prepare the manifests
---------------------

Training and validation each need a recording manifest and a segment manifest.
``LegacyAudioDataset`` uses the recording manifest to locate audio and the
segment manifest to associate an utterance with its speaker label.

Minimal recording manifest::

   id,storage_path,sample_freq,duration
   rec-0001,/absolute/path/to/rec-0001.wav,16000,7.2

Minimal segment manifest::

   id,recording,speaker,start,duration
   utt-0001,rec-0001,spk-001,0.0,3.5

The configured class file supplies the class inventory used by the classifier.
For speaker classification, it must contain the speaker ids referenced by the
segment manifests. Use the same class-file ordering for training and
validation, because it defines the classifier target indices.

Configuration layout
--------------------

Both training commands use a ``jsonargparse`` configuration file. The exact
available settings depend on the selected subcommand, so generate the argument
reference from the installed version before writing a configuration:

.. code-block:: bash

   hyperion-train-wav2xvector resnet1d --help
   hyperion-train-wav2vec2xvector hf_wav2vec2resnet1d --help

The native configuration has this high-level shape:

.. code-block:: yaml

   data:
     train:
       dataset:
         recordings_file: data/train/recordings.csv
         segments_file: data/train/segments.csv
         class_names: [speaker]
         class_files: [data/classes/speaker.csv]
         target_sample_freq: 16000
       sampler:
         # Batch/chunk sampling options from the selected sampler.
     val:
       dataset:
         recordings_file: data/val/recordings.csv
         segments_file: data/val/segments.csv
       sampler:
         # Validation sampling options.
   model:
     feats:
       audio_feats:
         sample_frequency: 16000
         audio_feat: logfb
     xvector:
       # Encoder, pooling, embedding, and classifier options for resnet1d.
   trainer:
     exp_path: exp/wav2xvector-resnet1d

The command links the training and validation ``class_files`` and data-loader
worker count. Define those shared values only in the training configuration
unless the parser help for your installed version says otherwise.

Train the native model
----------------------

Run the selected subcommand with the configuration file:

.. code-block:: bash

   hyperion-train-wav2xvector resnet1d --cfg configs/wav2xvector-resnet1d.yaml

The trainer writes its resolved configuration to
``<trainer.exp_path>/config.yaml`` and resumes from the most recent checkpoint
when one is available. Set an experiment path that is unique to the model and
data split; do not reuse it for an incompatible architecture or class list.

To launch through distributed PyTorch tooling, use the same command under your
site's ``torchrun`` invocation. Hyperion initializes distributed execution from
the standard local-rank environment.

Use a Wav2Vec2 x-vector model
-----------------------------

The data and trainer structure is the same, but the model block has an
``hf_feats`` section for the Hugging Face encoder and an ``xvector`` section
for the Hyperion head:

.. code-block:: bash

   hyperion-train-wav2vec2xvector hf_wav2vec2resnet1d \
     --cfg configs/hf-wav2vec2-resnet1d.yaml

Before training, verify that the configured Hugging Face model can be resolved
in the execution environment. Cache or mirror weights when training runs on
machines without internet access. Treat the encoder sample-frequency and the
dataset's ``target_sample_freq`` as one compatibility contract.

Extract embeddings
------------------

After training, use the saved model checkpoint with the matching waveform
extractor. Both extractors accept a ``HyperDataset`` manifest or separate
recording/segment inputs and write an embedding archive/specifier.

.. code-block:: bash

   hyperion-extract-wav2xvectors \
     --dataset-path data/eval/dataset.yaml \
     --model-path exp/wav2xvector-resnet1d/<checkpoint> \
     --xvector-path ark,csv:exp/eval/xvectors.ark,exp/eval/xvectors.csv \
     --use-gpu

For a Wav2Vec2-family checkpoint, substitute
``hyperion-extract-wav2vec2xvectors``. The extractor resamples audio to the
model's expected sample frequency and can use VAD entries from a
``HyperDataset`` manifest.

Validate the result
-------------------

Check that every expected input segment has an embedding key and that the
embedding dimension is consistent. Then create an enrollment map and trial key
for the evaluation set, score the vectors with cosine or PLDA, and evaluate
the result. The trial and score alignment rules are described in
:doc:`../data-model` and :doc:`../trials`.

Common problems
---------------

* **Class-index mismatch:** training and validation use different speaker
  class files or class ordering. Use one shared class inventory.
* **Sample-rate mismatch:** the dataset is not resampled to the native
  frontend or pretrained encoder's expected rate. Set
  ``target_sample_freq`` deliberately.
* **Out-of-memory failures:** lower the sampler's maximum batch duration or
  batch size before reducing model capacity.
* **Resume failure:** an experiment directory contains checkpoints for a
  different model shape or class list. Start with a new ``trainer.exp_path``.

See also
--------

* :doc:`../quickstart`
* :doc:`../data-model`
* :doc:`../torch`
* :doc:`extract-score-xvectors`
