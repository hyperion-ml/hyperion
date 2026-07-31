Use Configuration Files
=======================

Hyperion's maintained commands use ``jsonargparse``. A YAML or JSON file keeps
the data, model, trainer, and runtime choices for one experiment together,
while command-line options remain useful for small deliberate overrides.

Pass a configuration file
-------------------------

Commands that support a configuration file expose ``--cfg``. With commands
that use model subcommands, place the subcommand before ``--cfg``:

.. code-block:: bash

   hyperion-train-wav2xvector resnet1d \
     --cfg configs/wav2xvector-resnet1d.yaml

   hyperion-train-wav2vec2xvector hf_wav2vec2resnet1d \
     --cfg configs/hf-wav2vec2-resnet1d.yaml

Inspect the exact schema accepted by the installed command before creating a
new file:

.. code-block:: bash

   hyperion-train-wav2xvector resnet1d --help

Nested configuration mirrors parser groups
------------------------------------------

Nested parser groups become nested YAML mappings. For waveform x-vector
training, the principal sections are ``data``, ``model``, and ``trainer``:

.. code-block:: yaml

   data:
     train:
       dataset:
         recordings_file: data/train/recordings.csv
         segments_file: data/train/segments.csv
         class_names: [speaker]
         class_files: [data/classes/speaker.csv]
       sampler:
         sampler_type: class_weighted_random_seg_chunk_sampler
         min_chunk_length: 2.0
         max_batch_length: 120.0
     val:
       dataset:
         recordings_file: data/val/recordings.csv
         segments_file: data/val/segments.csv
       sampler:
         min_chunk_length: 2.0
         max_batch_length: 120.0
   model:
     feats:
       audio_feats:
         sample_frequency: 16000
         audio_feat: logfb
     xvector:
       # Architecture-specific fields; see the selected subcommand's help.
   trainer:
     exp_path: exp/wav2xvector-resnet1d-v1
     use_amp: true

YAML uses snake_case keys corresponding to the command's hyphenated options.
For example, ``--target-sample-freq`` becomes ``target_sample_freq`` and
``--exp-path`` becomes ``exp_path`` inside its nested group.

Use one canonical experiment configuration
------------------------------------------

Keep the complete configuration with its output directory. Waveform training
commands save the resolved configuration as ``<trainer.exp_path>/config.yaml``;
that file records the actual parsed values, including defaults.

Treat these values as one compatibility unit:

* model architecture and embedding dimensions;
* training and validation class CSV files and their ordering;
* audio sample rate and frontend/pretrained encoder choice;
* trainer optimizer, scheduler, AMP, and distributed settings.

Changing one of these after checkpoints exist should normally mean creating a
new ``trainer.exp_path``. See :doc:`run-resumable-distributed-training` for
resume behavior.

Override sparingly
------------------

Use a command-line option for ephemeral changes such as a new experiment path
or a temporary log level. For nested options, use the parser's dotted path:

.. code-block:: bash

   hyperion-train-wav2xvector resnet1d \
     --cfg configs/wav2xvector-resnet1d.yaml \
     --trainer.exp-path exp/wav2xvector-resnet1d-debug \
     --verbose 2

After a successful run, retain the resolved configuration written in the new
experiment directory rather than assuming that the source YAML captures every
default or override.

Common configuration failures
-----------------------------

* **Unknown key or option:** the key is misspelled, belongs to a different
  subcommand, or was renamed. Check that subcommand's ``--help`` output.
* **Incorrect nesting:** a key such as ``recordings_file`` is placed at the
  top level rather than under ``data.train.dataset`` or ``data.val.dataset``.
* **Linked-value conflict:** training and validation class files or worker
  counts conflict despite being linked by the command parser. Define the shared
  value once in the training section.
* **Resume incompatibility:** a configuration now describes a different model
  or target inventory from the checkpoint in ``exp_path``. Use a new path.

See also
--------

* :doc:`train-waveform-xvector`
* :doc:`train-pretrained-wav2vec2-xvector`
* :doc:`run-resumable-distributed-training`
