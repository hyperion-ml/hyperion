PyTorch Models and Tools
========================

The module ``hyperion.torch`` provides utilities, dataloaders, neural architectures and models based on PyTorch


Layers
------

These include several custom neural network layers.

Activation Function Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These includes a factory class the creates activation layers from config parameters, and custom activation layers.

.. autoclass:: hyperion.torch.layers.activation_factory.ActivationFactory

.. autoclass:: hyperion.torch.layers.swish.Swish

Normalization Layers
~~~~~~~~~~~~~~~~~~~~

These includes a factory class the creates normalizaton layers from config parameters.

.. autoclass:: hyperion.torch.layers.norm_layer_factory.NormLayerFactory

Dropout Layers
~~~~~~~~~~~~~~

These include custom dropout and drop-connect layers

.. automodule:: hyperion.torch.layers.dropout


Attention Layers
~~~~~~~~~~~~~~~~

Attention layers like the ones used in Transformers and Conformers.

.. automodule:: hyperion.torch.layers.attention

Pooling Layers
~~~~~~~~~~~~~~~

These include custom pooling layers and factory class to create pooling layers from config parameters.

.. automodule:: hyperion.torch.layers.pool_factory

.. automodule:: hyperion.torch.global_pool


Acoustic Feature Extraction Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These define several feature extraction layers that take wave as input and produce Spectrograms, Filter-banks, MFCC, etc. It also includes a factory class to create feature extraction layers from config params.

.. autoclass:: hyperion.torch.layers.audio_feats_factory.AudioFeatsFactory
	       
.. automodule:: hyperion.torch.layers.audio_feats

Feature Normalization Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hyperion.torch.layers.mvn

Feature Augmentation Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hyperion.torch.layers.spec_augment

Large Margin Losses Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~

These are output layers that are used to create large margin cross-entorpy losses.

.. automodule:: hyperion.torch.layers.margin_losses

Prob Densitiy Function Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are layers related to probability density functions used in VAEs

.. automodule:: hyperion.torch.layers.pdf_storage

.. automodule:: hyperion.torch.layers.tensor2pdf

Vector Quantization Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~

These are vector quantization layers like the ones used in VQ-VAEs

.. automodule:: hyperion.torch.layers.vq

Upsampling Layers
~~~~~~~~~~~~~~~~~~

These include layers related to upsampling operations.

.. automodule:: hyperion.torch.layers.interpolate

.. automodule:: hyperion.torch.layers.subpixel_convs


Positional Encoders
~~~~~~~~~~~~~~~~~~~

These include layers that implement positional encoders used in transformers.

.. automodule:: hyperion.torch.layers.pos_encoder

Calibration
~~~~~~~~~~~

These are layers that are used to simulate the calibration block after the speaker recognition back-end

.. automodule:: hyperion.torch.layers.calibrators


Layer Blocks
------------

These are Torch modules that combine several layers. These are the building blocks used to create more complex architectures like ResNets, Transformers of EfficientNets.

Fully Connected Blocks
~~~~~~~~~~~~~~~~~~~~~~

These are fully connected blocks used to create simple feed forward networks, classification heads, etc.

.. automodule:: hyperion.torch.layers.fc_blocks


Deep Convolutional Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deep Convolutional 1d Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are blocks to create deep convolutional networks 1d without residuals.

.. automodule:: hyperion.torch.layers.dc1d_blocks


Deep Convolutional 2d Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are blocks to create deep convolutional networks 2d without residuals.

.. automodule:: hyperion.torch.layers.dc2d_blocks

TDNN Blocks
~~~~~~~~~~~

TDNN Blocks
^^^^^^^^^^^

TDNN blocks used to create TDNN x-vectors

.. automodule:: hyperion.torch.layers.tdnn_blocks

Extended TDNN Blocks
^^^^^^^^^^^^^^^^^^^^

Extended TDNN blocks used to create E-TDNN x-vectors

.. automodule:: hyperion.torch.layers.etdnn_blocks

Residual Extended TDNN Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extended TDNN blocks with residual connections

.. automodule:: hyperion.torch.layers.resetdnn_blocks

Squeeze-Excitation Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~

Squeeze-Excitation Blocks 1d and 2d, which are added at the output ResNet blocks and other to create squeeze-excitation networks.

.. automodule:: hyperion.torch.layers.se_blocks

Cannonical ResNet Blocks
~~~~~~~~~~~~~~~~~~~~~~~~

These are blocks used to create cannonical ResNet, SE-ResNet, Res2Nets, etc.

ResNet Blocks
^^^^^^^^^^^^^

These blocks are used to create cannonical ResNets.

.. automodule:: hyperion.torch.layers.resnet_blocks

SE-ResNet Blocks
^^^^^^^^^^^^^^^^

These blocks are used to create cannonical Squeeze-Excitation ResNets

.. automodule:: hyperion.torch.layers.seresnet_blocks

SE-ResNet Blocks
^^^^^^^^^^^^^^^^

These blocks are used to create cannonical Squeeze-Excitation ResNets.

.. automodule:: hyperion.torch.layers.res2net_blocks


SpineNet Blocks
~~~~~~~~~~~~~~~

These are some extra blocks needed to build SpineNet and Spine2Net.

.. automodule:: hyperion.torch.layers.spine_blocks

MobileNet Blocks
~~~~~~~~~~~~~~~~

These are blocks needed to build EfficientNet networks.

.. automodule:: hyperion.torch.layers.mbconv_blocks


Generic ResNet Blocks
~~~~~~~~~~~~~~~~~~~~~

ResNet 1d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible ResNets based on 1d convs.

.. automodule:: hyperion.torch.layers.resnet1d_blocks

Res2Net 1d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible Res2Nets based on 1d convs.

.. automodule:: hyperion.torch.layers.res2net1d_blocks

ResNet 2d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible ResNets based on 2d convs.

.. automodule:: hyperion.torch.layers.resnet2d_blocks

Res2Net 2d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible Res2Nets based on 2d convs.

.. automodule:: hyperion.torch.layers.res2net2d_blocks


Transformer Blocks
~~~~~~~~~~~~~~~~~~

These are blocks used to build Transformers.

.. automodule:: hyperion.torch.layers.transformer_conv2d_subsampler

.. automodule:: hyperion.torch.layers.transformer_encoder_v1

.. automodule:: hyperion.torch.layers.transformer_feedforward

Conformer Blocks
~~~~~~~~~~~~~~~~

.. automodule:: hyperion.torch.layers.conformer_encoder_v1

.. automodule:: hyperion.torch.layers.conformer_conv

		
Torch Models and Model Loader
-----------------------------

All PyTorch ML Neural Architectures and Models in Hyperion derive from the same base class

.. autoclass:: hyperion.torch.TorchModel

The ``TorchModelLoader`` can load any model or network architecture from file.

.. autoclass:: hyperion.torch_model_loader.TorchModelLoader


Neural Architectures
--------------------

All neural architectures derive from the ``NetArch`` class.

.. autoclass:: hyperion.torch.narchs.net_arch.NetArch

The ``TorchNALoader`` can load any network architecture from file.

.. autoclass:: hyperion.torch.narchs.torch_na_loader.TorchNALoader

Acoustic Features
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.audio_feats_mvn.AudioFeatsMVN

Fully Connected Network
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.fc_net.FCNet

Classification Head
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.classif_head.ClassifHead


Deep Convolutional Encoder/Decoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are Encoder/Decoders based on Deep Convolutional Networks 1d and 2d.

DC Encoder 1d
^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.dc1d_encoder.DC1dEncoder

DC Decoder 1d
^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.dc1d_decoder.DC1dDecoder

DC Encoder 2d
^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.dc2d_encoder.DC2dEncoder

DC Decoder 2d
^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.dc2d_decoder.DC2dDecoder


TDNN Variants
~~~~~~~~~~~~~

These are variants of TDNNs. There is a factory class that creates TDNN networks from config params.

.. autoclass:: hyperion.torch.narchs.tdnn_factory.TDNNFactory

TDNN
^^^^

.. autoclass:: hyperion.torch.narchs.tdnn.TDNN

E-TDNN
^^^^^^

.. autoclass:: hyperion.torch.narchs.etdnn.ETDNN

Residual E-TDNN
^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.resetdnn.ResETDNN


Cannonical ResNets/SE-ResNets/Res2Nets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These classes can be used to build cannonical ResNets, SE-ResNets and Res2Nets.
There is a factory class that creates ResNets from config params.

.. autoclass:: hyperion.torch.narchs.resnet_factory.ResNetFactory

.. automodule:: hyperion.torch.narchs.resnet


SpineNets/Spine2Nets
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.spinenet_factory.SpineNetFactory

.. automodule:: hyperion.torch.narchs.spinenet

ResNet Encoder/Decoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are Encoder/Decoders based on flexible ResNets 1d and 2d.

ResNet Encoder 1d
^^^^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.resnet1d_encoder.ResNet1dEncoder

ResNet Decoder 1d
^^^^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.resnet1d_decoder.ResNet1dDecoder

ResNet Encoder 2d
^^^^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.resnet2d_encoder.ResNet2dEncoder

ResNet Decoder 2d
^^^^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.resnet2d_decoder.ResNet2dDecoder


EfficientNet
~~~~~~~~~~~~

.. autoclasss:: hyperion.torch.narchs.efficient_net.EfficientNet


Transformer
~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.transformer_encoder_v1.TransformerEncoderV1


Conformer
~~~~~~~~~~~

.. autoclass:: hyperion.torch.narchs.transformer_encoder_v1.ConformerEncoderV1


Models
------

Losses
------


Adversarial Attacks
-------------------

Trainers
--------

Datasets, Data Loaders and Samplers
-----------------------------------


Data Transformations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.transforms.reshape.Reshape

Optimizers
----------

Learning Rate Schedulers
------------------------


Loggers
-------


Utils
-----
