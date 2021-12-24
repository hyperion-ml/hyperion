.. hyperion documentation master file, created by
   sphinx-quickstart on Mon Mar 16 16:33:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hyperion: Speaker Recognition Toolkit
====================================
..
   .. image:: ./image/hyperion_logo.png

Hyperion is a Speaker Recognition Toolkit based on PyTorch and numpy. It provides:
 * x-Vector architectures: ResNet, Res2Net, Spine2Net, ECAPA-TDNN, EfficientNet, Transformers and others.
 * Embedding preprocessing tools: PCA, LDA, NAP, Centering/Whitening, Length Normalization, CORAL
 * Several flavours of PLDA back-ends: Full-rank PLDA, Simplified PLDA, PLDA
 * Calibration and Fusion tools
 * Recipes for popular datasets: VoxCeleb, NIST-SRE, VOiCES

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   getting-started.rst
   numpy.rst
   torch.rst
   io.rst
   utils.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
