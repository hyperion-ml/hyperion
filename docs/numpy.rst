Numpy Models and Tools
======================

Hyperion provides several models and feature extractors based on numpy.

Feature Extraction and Voice Activity Detection
-----------------------------------------------

Feature Extraction Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hyperion.feats.mfcc

.. autoclass:: hyperion.feats.filter_banks.FilterBankFactory
	       
.. autoclass:: hyperion.feats.feature_windows.FeatureWindowFactory
	     
Feature Normalization Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hyperion.feats.feature_normalization

Voice Activity Detection Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.feats.energy_vad.EnergyVAD

Feature Extraction Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hyperion.feats.stft


Speech Augmentation
-------------------

Combined Speech Augmentation Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.augment.speech_augment.SpeechAugment

Noise Augmentation Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.augment.noise_augment.NoiseAugment

.. autoclass:: hyperion.augment.noise_augment.SingleNoiseAugment


Reverberation Augmentation Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.augment.reverb_augment.ReverbAugment

.. autoclass:: hyperion.augment.reverb_augment.SingleReverbAugment

Speed Augmentation Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.augment.speed_augment.SpeedAugment


Hyperion Numpy Models
---------------------

All numpy ML models in Hyperion derive from the same base class

.. autoclass:: hyperion.hyp_model.HypModel

.. autoclass:: hyperion.model_loader.ModelLoader
	       

Probability Density Functions
-----------------------------

These are classes that define different probability density functions like GMMs and PLDA

Core PDF Classes
~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.pdfs.core.pdf.PDF

.. autoclass:: hyperion.pdfs.core.exp_family.ExpFamily

.. autoclass:: hyperion.pdfs.core.normal_cov.NormalCov
	       
.. autoclass:: hyperion.pdfs.core.normal_diag_cov.NormalDiagCov


PLDA Classes
~~~~~~~~~~~~

.. autoclass:: hyperion.pdfs.plda.plda_base.PLDABase

.. autoclass:: hyperion.pdfs.plda.frplda.FRPLDA

.. autoclass:: hyperion.pdfs.plda.splda.SPLDA

.. autoclass:: hyperion.pdfs.plda.plda.PLDA

	       
Mixture Models
~~~~~~~~~~~~~~

.. autoclass:: hyperion.pdfs.mixtures.exp_family_mixture.ExpFamilyMixture

.. autoclass:: hyperion.pdfs.mixtures.gmm.GMM

.. autoclass:: hyperion.pdfs.mixtures.gmm_diag_cov.GMMDiagCov

.. autoclass:: hyperion.pdfs.mixtures.gmm_tied_diag_cov.GMMTiedDiagCov


Classifiers and Calibrators
---------------------------

Gaussian Classifiers
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.classifiers.linear_gbe.LinearGBE

SVM Classifiers
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.classifiers.linear_svmc.LinearSVMC
	     

Logistic Regression Classifiers and Calibrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.classifiers.logistic_regression.LogisticRegression

.. autoclass:: hyperion.classifiers.binary_logistic_regression.BinaryLogisticRegression


Clustering Tools
----------------

.. autoclass:: hyperion.clustering.kmeans.KMeans

.. autoclass:: hyperion.clustering.ahc.AHC

	       
Score Normalization
-------------------

.. autoclass:: hyperion.score_norm.score_norm.ScoreNorm

.. autoclass:: hyperion.score_norm.t_norm.TNorm

.. autoclass:: hyperion.score_norm.z_norm.ZNorm

.. autoclass:: hyperion.score_norm.zt_norm.ZTNorm

.. autoclass:: hyperion.score_norm.tz_norm.TZNorm

.. autoclass:: hyperion.score_norm.s_norm.SNorm

.. autoclass:: hyperion.score_norm.adapt_s_norm.AdaptSNorm


Feature Transformations
-----------------------

These are classes to apply feature transformations/projections like PCA, LDA, etc.

Transform Classes
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.transforms.pca.PCA

.. autoclass:: hyperion.transforms.lda.LDA

.. autoclass:: hyperion.transforms.cent_whiten.CentWhiten

.. autoclass:: hyperion.transforms.lnorm.LNorm

.. autoclass:: hyperion.transforms.coral.CORAL

.. autoclass:: hyperion.transforms.gaussianizer.Gaussianizer

.. autoclass:: hyperion.transforms.nap.NAP

.. autoclass:: hyperion.transforms.nda.NDA

.. autoclass:: hyperion.transforms.mvn.MVN

.. autoclass:: hyperion.transforms.skl_tsne.SklTSNE


Sequence of Transformations Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.transforms.transform_list.TransformList

Auxiliary Classes
~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.transforms.sb_sw.SbSw


Metrics
-------

Metric Functions
~~~~~~~~~~~~~~~~

These are some functions to compute performance metrics used in speaker identification and verification

.. automodule:: hyperion.metrics.eer

.. automodule:: hyperion.metrics.dcf

.. automodule:: hyperion.metrics.cllr

.. automodule:: hyperion.metrics.roc

.. automodule:: hyperion.metrics.acc

.. automodule:: hyperion.metrics.confusion_matrix

.. automodule:: hyperion.metrics.utils
		

Helper Code Blocks
------------------

Classes and codeblocks that are re-used in several scripts

.. autoclass:: hyperion.helpers.vector_reader.VectorReader

.. autoclass:: hyperion.helpers.vector_class_reader.VectorClassReader

.. autoclass:: hyperion.helpers.trial_data_reader.TrialDataReader

.. autoclass:: hyperion.helpers.multi_test_trial_data_reader.MultiTestTrialDataReader

.. autoclass:: hyperion.helpers.multi_test_trial_data_reader_v2.MultiTestTrialDataReaderV2

.. autoclass:: hyperion.helpers.plda_factor.PLDAFactory
	       
