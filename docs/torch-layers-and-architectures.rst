PyTorch Layers and Architecture Catalog
=======================================

This catalog helps select reusable PyTorch components before writing a new task
model. It is a map of supported package exports, not a requirement to import
every listed class directly. Use the nearest compatible architecture or factory
first; a new task model should compose these pieces and retain the serialization
contract in :doc:`torch-extension-points`.

Input representation
--------------------

Choose components according to the representation at the point where they are
used:

* waveforms are generally shaped ``(batch, num_samples)``;
* frame sequences generally have a time and feature/channel axis, with the
  exact layout defined by their model or frontend; and
* 2-D acoustic inputs carry channel, frequency, and time axes.

Pass valid lengths or masks through the model whenever the implementation
supports them. Do not infer padding from zero-valued samples or frames: zero
can be valid audio or an ordinary feature value.

Frontends, normalization, and pooling
-------------------------------------

.. autoclass:: hyperion.torch.layers.audio_feats_factory.AudioFeatsFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.layers.norm_layer_factory.NormLayer1dFactory
   :no-index:
   :members: create

.. autoclass:: hyperion.torch.layers.pool_factory.GlobalPool1dFactory
   :no-index:
   :members: create, filter_args, add_class_args

``AudioFeatsFactory`` creates waveform-to-feature operations such as
log-filterbanks and MFCCs. Normalization and pooling factories are the
configuration-facing choices for architectures; they avoid hard-coding a
normalization or utterance-pooling policy into a task model. The exported
pooling layers include mean, mean/std, attention-weighted, log-variance, and
LDE-style variants.

For pretrained transformer frontends, use the wrappers described in
:doc:`torch-integrations-and-robustness`, rather than combining a raw external
model with internal pooling code.

Reusable blocks
---------------

``hyperion.torch.layer_blocks`` provides compositions used by architecture
classes. The main families are:

* TDNN and extended TDNN blocks for frame sequences;
* ResNet/Res2Net, squeeze-excitation, and ConvNeXt blocks for 1-D and 2-D
  encoders or decoders;
* Conformer and Transformer encoder blocks for context-aware sequences;
* FC, MBConv, SpineNet, and DC encoder/decoder blocks for specialised
  architectures; and
* projection, classifier, and Hydra heads when a model needs one or more task
  outputs.

Layer blocks are implementation-level ``torch.nn.Module`` components. Reuse
their accepted tensor layout and constructor options from the architecture that
already uses them. A block is not usually a serializable deployment artifact by
itself.

Neural architecture families
----------------------------

The following architecture families are stable building blocks for conventional
speech and speaker-recognition models:

``TDNNV1`` and ``ETDNNV1``
  Frame-sequence encoders. Their respective factories expose maintained
  configuration selections.

``ResNet``, ``ResNet1dEncoder``, and ``ResNet2dEncoder``
  Residual encoders for one- or two-dimensional acoustic representations.
  The ResNet factory selects the supported named variants.

``ConformerEncoderV1`` and ``TransformerEncoderV1``/``TransformerEncoderV2``
  Context-aware sequence encoders. Specify attention context and mask/length
  behavior consistently with the enclosing model.

``ConvNext1dEncoder``, ``ConvNext2dEncoder``, ``EfficientNet``, and ``SpineNet``
  Convolutional alternatives with their associated factory/configuration
  interfaces.

``ClassifHead``, ``ProjHead``, ``HydraHead``, and ``FeatFuserMVN``
  Output, multi-task, projection, and feature-fusion components. Select them
  in the task model, where loss and target semantics are known.

All architecture classes derive from the shape-reporting contract described in
:doc:`torch-extension-points`. A model that combines an encoder with a pooling
and classification head should document the intermediate layout at each
boundary, especially when it exposes embedding extraction.

Factories and selection
-----------------------

.. autoclass:: hyperion.torch.narchs.resnet_factory.ResNetFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.narchs.tdnn_factory.TDNNFactory
   :no-index:
   :members: create, filter_args, add_class_args

.. autoclass:: hyperion.torch.narchs.spinenet_factory.SpineNetFactory
   :no-index:
   :members: create, filter_args, add_class_args

Use a factory at a CLI/configuration boundary, then save the resolved model
configuration with the experiment. Factories centralize aliases and valid
arguments; copying their selection logic into a command makes configuration
compatibility harder to preserve.

Experimental architecture families
----------------------------------

DAC/streaming-DAC layers and architectures, transducer decoder/predictor
blocks, and vector-quantization layers used by q-vector or codec workflows are
experimental in those contexts. They are described in
:doc:`experimental-components`; do not assume their architecture names,
checkpoint compatibility, or output-code semantics are stable.

See also
--------

* :doc:`torch-api`
* :doc:`torch-extension-points`
* :doc:`experimental-components`
