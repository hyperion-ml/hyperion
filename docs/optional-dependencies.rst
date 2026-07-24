Optional Dependencies, External Assets, and TPM
================================================

Hyperion has a broad Python package surface, but an individual workflow should
only require the runtime it actually uses. This page defines what the CLI
documentation means by a *conditional runtime requirement* and how the stable
third-party model (TPM) wrappers behave.

Requirement levels
------------------

The command index labels a requirement when it is needed **for that command**;
it does not mean every Hyperion user must install it. There are four levels:

* **Core package runtime:** normal package dependencies, including PyTorch,
  audio IO, and ``jsonargparse``. Select the PyTorch/CUDA extra for the host;
  see :doc:`getting-started`.
* **Project extra:** an explicitly optional package extra. VoxProfile is the
  current example: install ``pip install -e .[voxprofile]`` before using its
  evaluator or CLI command.
* **External model asset:** a checkpoint, ONNX model, tokenizer, or corpus
  resource that the user must provide or allow the wrapper to retrieve.
* **Remote retrieval:** a wrapper may download an asset on first use. This is
  an observable network and cache side effect, not a transparent fallback.

When a command needs any non-core level, its inventory entry and generated
reference say ``Conditional runtime requirements``. Install/provision them
before running that command. A command that does not use the dependency remains
usable without it.

Offline and reproducible operation
----------------------------------

For production or offline runs, provision every external asset before the job.
Pass local paths whenever the parser exposes them, retain the asset file and
checksum, and record package versions, model identifier/revision, cache
location, device, and complete CLI configuration. Do not rely on a mutable
remote default or an already-populated personal cache.

The Sphinx build deliberately mocks heavyweight dependencies. It validates
documentation structure, not that external models can be imported, downloaded,
or executed. The generated CLI option reference instead runs real ``--help``
entry points and reports a diagnostic if a parser cannot import. See
:doc:`building-documentation`.

TPM wrappers
------------

TPM wrappers are stable Hyperion interfaces, but their output also depends on a
third-party package and asset. API stability does not guarantee that a remote
model revision, license, model quality, or cache contents remain unchanged.

Hugging Face frontends
~~~~~~~~~~~~~~~~~~~~~~

Hugging Face Wav2Vec2/Hubert/WavLM/Whisper wrappers require ``transformers``.
``pretrained_model_path`` may be a local directory or a Hugging Face identifier.
For an identifier, the wrapper can retrieve artifacts into ``cache_dir``;
``force_download`` requests a fresh retrieval. Pin ``revision`` to an immutable
revision and use a local pre-populated model path for offline jobs.

DNSMOS
~~~~~~

DNSMOS requires an ONNX Runtime build compatible with the requested device.
With no supplied ``primary_model_path`` or ``p808_model_path``, it downloads
the corresponding ONNX file into ``cache_dir`` (default
``~/.cache/hyperion/dnsmos``). Provide local model paths to prevent retrieval.
The primary P.835 model is required. The optional P.808 regressor may be
disabled; if it cannot load, DNSMOS logs a warning and continues without its
P.808 score. Reports must say whether P.808 was enabled and identify the model
files used.

UTMOS v2
~~~~~~~~

UTMOS v2 requires the external ``utmosv2`` runtime. On first scoring it asks
that runtime for its pretrained model and writes temporary 16 kHz audio files
under ``tmp_dir`` (default ``./utmos_cache`` with a process-specific child
directory). Ensure that location is writable, has sufficient space, and is
handled according to the audio-data retention policy. Keep the UTMOS runtime
and model version with reported MOS values.

VoxProfile
~~~~~~~~~~

VoxProfile requires the ``voxprofile`` project extra plus the model assets
selected by evaluator/CLI arguments. Supply local checkpoint paths and record
the model set, device, and asset licenses. A missing VoxProfile extra or asset
is an execution error; Hyperion does not substitute another profile model.

Command behavior
----------------

``hyperion-eval-speech-quality-metrics`` may enable DNSMOS and/or UTMOS v2.
``hyperion-eval-voxprofile-metrics`` requires the VoxProfile extra and selected
model paths. Wav2Vec2-family train, fine-tune, extraction, and logit commands
need ``transformers`` and compatible pretrained assets. Experimental codec,
VITS/FreeVC, transducer, and Q-vector commands additionally require matching
version-coupled checkpoints; see :doc:`cli/experimental`.

Before relying on any such command, run its ``--help`` in the target runtime,
then run a small local input through the complete workflow. A successful import
or download is not validation of the resulting model outputs.

See also
--------

* :doc:`torch-integrations-and-robustness`
* :doc:`cli`
* :doc:`getting-started`
