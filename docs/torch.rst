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

Upsampling Classes
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


Torch Models and Model Loader
-----------------------------


Neural Architectures
--------------------


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
