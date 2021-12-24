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

.. automodule:: hyperion.torch.layers.norm_layer_factory

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

.. automodule:: hyperion.torch.layers.global_pool


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

.. automodule:: hyperion.torch.layer_blocks.fc_blocks


Deep Convolutional Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deep Convolutional 1d Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are blocks to create deep convolutional networks 1d without residuals.

.. automodule:: hyperion.torch.layer_blocks.dc1d_blocks


Deep Convolutional 2d Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are blocks to create deep convolutional networks 2d without residuals.

.. automodule:: hyperion.torch.layer_blocks.dc2d_blocks

TDNN Blocks
~~~~~~~~~~~

TDNN Blocks
^^^^^^^^^^^

TDNN blocks used to create TDNN x-vectors

.. automodule:: hyperion.torch.layer_blocks.tdnn_blocks

Extended TDNN Blocks
^^^^^^^^^^^^^^^^^^^^

Extended TDNN blocks used to create E-TDNN x-vectors

.. automodule:: hyperion.torch.layer_blocks.etdnn_blocks

Residual Extended TDNN Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extended TDNN blocks with residual connections

.. automodule:: hyperion.torch.layer_blocks.resetdnn_blocks

Squeeze-Excitation Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~

Squeeze-Excitation Blocks 1d and 2d, which are added at the output ResNet blocks and other to create squeeze-excitation networks.

.. automodule:: hyperion.torch.layer_blocks.se_blocks

Cannonical ResNet Blocks
~~~~~~~~~~~~~~~~~~~~~~~~

These are blocks used to create cannonical ResNet, SE-ResNet, Res2Nets, etc.

ResNet Blocks
^^^^^^^^^^^^^

These blocks are used to create cannonical ResNets.

.. automodule:: hyperion.torch.layer_blocks.resnet_blocks

SE-ResNet Blocks
^^^^^^^^^^^^^^^^

These blocks are used to create cannonical Squeeze-Excitation ResNets

.. automodule:: hyperion.torch.layer_blocks.seresnet_blocks

SE-ResNet Blocks
^^^^^^^^^^^^^^^^

These blocks are used to create cannonical Squeeze-Excitation ResNets.

.. automodule:: hyperion.torch.layer_blocks.res2net_blocks


SpineNet Blocks
~~~~~~~~~~~~~~~

These are some extra blocks needed to build SpineNet and Spine2Net.

.. automodule:: hyperion.torch.layer_blocks.spine_blocks

MobileNet Blocks
~~~~~~~~~~~~~~~~

These are blocks needed to build EfficientNet networks.

.. automodule:: hyperion.torch.layer_blocks.mbconv_blocks


Generic ResNet Blocks
~~~~~~~~~~~~~~~~~~~~~

ResNet 1d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible ResNets based on 1d convs.

.. automodule:: hyperion.torch.layer_blocks.resnet1d_blocks

Res2Net 1d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible Res2Nets based on 1d convs.

.. automodule:: hyperion.torch.layer_blocks.res2net1d_blocks

ResNet 2d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible ResNets based on 2d convs.

.. automodule:: hyperion.torch.layer_blocks.resnet2d_blocks

Res2Net 2d Blocks
^^^^^^^^^^^^^^^^

These are blocks used to buld flexible Res2Nets based on 2d convs.

.. automodule:: hyperion.torch.layer_blocks.res2net2d_blocks


Transformer Blocks
~~~~~~~~~~~~~~~~~~

These are blocks used to build Transformers.

.. automodule:: hyperion.torch.layer_blocks.transformer_conv2d_subsampler

.. automodule:: hyperion.torch.layer_blocks.transformer_encoder_v1

.. automodule:: hyperion.torch.layer_blocks.transformer_feedforward

Conformer Blocks
~~~~~~~~~~~~~~~~

.. automodule:: hyperion.torch.layer_blocks.conformer_encoder_v1

.. automodule:: hyperion.torch.layer_blocks.conformer_conv

		
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

.. autoclass:: hyperion.torch.narchs.tdnn.TDNNV1

E-TDNN
^^^^^^

.. autoclass:: hyperion.torch.narchs.etdnn.ETDNNV1

Residual E-TDNN
^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.narchs.resetdnn.ResETDNNV1


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

.. autoclass:: hyperion.torch.narchs.conformer_encoder_v1.ConformerEncoderV1


Models
------

These include complex models created by connecting several network architectures.

x-Vectors
~~~~~~~~~

There are several variants of x-vector embeddings. They all derive from the same base class.

.. autoclass:: hyperion.torch.models.xvectors.xvector.XVector

TDNN x-Vector
^^^^^^^^^^^^^

x-Vectors with TDNN, E-TDNN, Residual E-TDNN Encoders.

.. autoclass:: hyperion.torch.models.xvectors.tdnn_xvector.TDNNXVector

ResNet x-Vector
^^^^^^^^^^^^^^^

x-Vectors with Cannonical ResNet, Res2Net Encoders.

.. autoclass:: hyperion.torch.models.xvectors.resnet_xvector.ResNetXVector


SpineNet x-Vector
^^^^^^^^^^^^^^^^^

x-Vectors with SpineNet, Spine2Net Encoders.

.. autoclass:: hyperion.torch.models.xvectors.spinenet_xvector.SpineNetXVector

ResNet 1d x-Vector
^^^^^^^^^^^^^^^^^^

x-Vectors with ResNet, Res2Net 1d Encoders. It can be cofigured as ECAPA-TDNN

.. autoclass:: hyperion.torch.models.xvectors.resnet1d_xvector.ResNet1dXVector


Transfomer x-Vector
^^^^^^^^^^^^^^^^^^^

x-Vectors based on Transformer Encoder

.. autoclass:: hyperion.torch.models.xvectors.transformer_xvector_v1.TransformerXVectorV1


Auto-Encoder
~~~~~~~~~~~~

.. autoclass:: hyperion.torch.models.ae.ae.AE


Variational Auto-Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.models.vae.vae.VAE

.. autoclass:: hyperion.torch.models.vae.vq_vae.VQVAE


Losses
------

Custom loss classes

.. autoclass:: hyperion.torch.losses.bce_with_llr.BCEWithLLR

Adversarial Attacks
-------------------

It contains classes to generate adversarial attacks for speaker recognition.

Attack Generation Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

All the adv. attacks derive from the same base class:

.. autoclass:: hyperion.torch.adv_attacks.adv_attack.AdvAttack

FGSM
^^^^

.. autoclass:: hyperion.torch.adv_attacks.fgsm_attack.FGSMAttack

.. autoclass:: hyperion.torch.adv_attacks.snr_fgsm_attack.SNRFGSMAttack

.. autoclass:: hyperion.torch.adv_attacks.rand_fgsm_attack.RandFGSMAttack

.. autoclass:: hyperion.torch.adv_attacks.iter_fgsm_attack.IterFGSMAttack

PGD
^^^

.. autoclass:: hyperion.torch.adv_attacks.pgd_attack.PGDAttack

Carlini-Wagner
^^^^^^^^^^^^^^

Carlini-Wagner attacks derive from the same base class:

.. autoclass:: hyperion.torch.adv_attacks.carlini_wagner.CarliniWagner

.. autoclass:: hyperion.torch.adv_attacks.carlini_wagner_l2.CarliniWagnerL2

.. autoclass:: hyperion.torch.adv_attacks.carlini_wagner_linf.CarliniWagnerLInf

.. autoclass:: hyperion.torch.adv_attacks.carlini_wagner_l0.CarliniWagnerL0


Attack Generator Factories
~~~~~~~~~~~~~~~~~~~~~~~~~~

These are factory classes that create attack generator objects. They create attacks from Hyperion or from the `Adversarial Robustness Toolbox <https://github.com/Trusted-AI/adversarial-robustness-toolbox>`

.. autoclass:: hyperion.torch.adv_attacks.attack_factory.AttackFactory

.. autoclass:: hyperion.torch.adv_attacks.random_attack_factory.RandomAttackFactory

.. autoclass:: hyperion.torch.adv_attacks.art_attack_factory.ARTAttackFactory

	       
Trainers
--------

Generic Trainer
~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.trainers.torch_trainer.TorchTrainer

x-Vector Trainers
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.trainers.xvector_trainer.XVectorTrainer

.. autoclass:: hyperion.torch.trainers.xvector_trainer_from_wav.XVectorTrainerFromWav
	       
.. autoclass:: hyperion.torch.trainers.xvector_trainer_deep_feat_reg.XVectorTrainerDeepFeatReg

.. autoclass:: hyperion.torch.trainers.xvector_trainer_deep_feat_reg_from_wav.XVectorTrainerDeepFeatRegFromWav


Auto-encoder Trainer
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.trainers.ae_trainer.AETrainer

VAE Trainers
~~~~~~~~~~~~

.. autoclass:: hyperion.torch.trainers.vae_trainer.VAETrainer

.. autoclass:: hyperion.torch.trainers.dvae_trainer.DVAETrainer

VQ-VAE Trainers
~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.trainers.vq_vae_trainer.VQVAETrainer

.. autoclass:: hyperion.torch.trainers.vq_dvae_trainer.VQDVAETrainer

	       
Datasets, Data Loaders and Samplers
-----------------------------------

Datasets
~~~~~~~~

Audio Datasets
^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.data.audio_dataset.AudioDataset

Feature Sequence Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.data.feat_seq_dataset.FeatSeqDataset

.. autoclass:: hyperion.torch.data.paired_feat_seq_dataset.PairedFeatSeqDataset

Embedding Datasets
^^^^^^^^^^^^^^^^^^

.. autoclass:: hyperion.torch.data.embed_dataset.EmbedDataset


Samplers
~~~~~~~~

.. automodule:: hyperion.torch.data.weighted_seq_sampler

.. automodule:: hyperion.torch.data.weighted_embed_sampler

Data Transformations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.transforms.reshape.Reshape

Optimizers
----------

These are custom optimizers and a factory class to create optimizers from config params.

Custom Optimizers
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.optim.radam.RAdam

Optimizer Factory
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.optim.factory.OptimizerFactory
	       

Learning Rate Schedulers
------------------------

These are custom learning rate schedulers and a factory class to create schedulers from config params.

Custom LR Schedulers
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.lr_schedulers.red_lr_on_plateau.ReduceLROnPlateau

.. autoclass:: hyperion.torch.lr_schedulers.exp_lr.ExponentialLR

.. autoclass:: hyperion.torch.lr_schedulers.invpow_lr.InvPowLR

.. autoclass:: hyperion.torch.lr_schedulers.cos_lr.CosineLR
	       

LR Scheduler Factory
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.lr_schedulers.factory.LRSchedulerFactory


Metrics
-------

This are metric classes and functions that cannot be used as loss function.

Metric Classes
~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.metrics.metrics.TorchMetric
	       
.. automodule:: hyperion.torch.metrics.accuracy

Metric Functions
~~~~~~~~~~~~~~~~
.. automodule:: hyperion.torch.metrics.accuracy_functional

Loggers
-------

The logger classes are used to write information to standard output, log files, tensorboard or WandB.
The LoggerList class contains a set of loggers. When we log something to the LoggerList, the same is written
in all the loggers contained in it. The loggers support multi-gpu training with ``DistributedDataParallel``

Individual Loggers
~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.loggers.logger.Logger

.. autoclass:: hyperion.torch.loggers.prog_logger.ProgLogger

.. autoclass:: hyperion.torch.loggers.csv_logger.CSVLogger

.. autoclass:: hyperion.torch.loggers.tensorboard_logger.TensorBoardLogger

.. autoclass:: hyperion.torch.loggers.wandb_logger.WAndBLogger

Logger List
~~~~~~~~~~~

.. autoclass:: hyperion.torch.loggers.logger_list.LoggerList

Utils
-----

Device Handling Utils
~~~~~~~~~~~~~~~~~~~~~

Utilities to handle GPU devices, like finding a free GPU in a shared server.

.. automodule:: hyperion.torch.utils.devices

Distributed Data Parallel Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These contains utils to perform multigpu training with Distributed Data Paralell.

.. automodule:: hyperion.torch.utils.ddp

Metric Accumulators
~~~~~~~~~~~~~~~~~~~

Tools to combine the metrics computed in multiple GPUs into a single metric

.. automodule:: hyperion.torch.utils.metric_acc

Evaluation Utils
~~~~~~~~~~~~~~~~

Functions that can be usefull when evaluating neural networks. For example, when a signal is too long to fit
in memory and needs to be splitted into chunks

.. automodule:: hyperion.torch.utils.eval_utils

Math Functions
~~~~~~~~~~~~~~

.. automodule:: hyperion.torch.utils.math


Miscellaneous Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hyperion.torch.utils.misc

