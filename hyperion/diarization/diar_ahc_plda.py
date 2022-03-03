"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from ..clustering import AHC
from ..pdfs import GMMTiedDiagCov as GMM
from ..transforms import PCA, LNorm


class DiarAHCPLDA(object):
    """Class to perform diarization using
    Agglomerative clustering using scores computed by a PLDA model.

    The steps are:
    - It applies a pre-processing transformation to the data, such as LDA and
      Length normalization (optional).
    - Trains PCA on the test data and reduces test data dimension. It also
      transforms the parameters of the PLDA model using the PCA projection matrix (optional).
    - Gets affinity matrix using PLDA scoring.
    - It applies unsupervised calibration to scores using GMM model (optional).
    - Performs AHC.

    Attributes:
      plda_model: pre-trained PLDA model.
      preproc: preprocessing transformation class.
               If None, no transformation is applied.
      threshold: stopping threshold for AHC.
      pca_var_r: ratio of variance to keep when doing PCA on features after
                 the preprocessing. If "pca_var_r=1", PCA is not applied.
      do_unsup_cal: applies unsupervised calibration to PLDA scores.
      use_bic: uses Bayesian Information Criterion to decide if there is 1 or 2 components
               in the GMM used for calibration.
    """

    def __init__(
        self,
        plda_model,
        preproc=None,
        threshold=0,
        pca_var_r=1,
        do_unsup_cal=False,
        use_bic=False,
    ):

        self.plda_model = plda_model
        self.preproc = preproc
        self.threshold = threshold
        self.pca_var_r = pca_var_r
        self.do_unsup_cal = do_unsup_cal
        self.use_bic = use_bic and do_unsup_cal
        self._ahc = AHC()

    @staticmethod
    def _plot_score_hist(scores, output_file, thr=None, gmm=None):
        """Plots the score histograms and GMM."""
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        mask = np.triu(np.ones(scores.shape, dtype=np.bool), 1)
        scores_r = scores[mask].ravel()

        _, bins, _ = plt.hist(
            scores_r,
            100,
            histtype="step",
            density=True,
            color="b",
            linestyle="solid",
            linewidth=1.5,
        )

        if thr is not None:
            plt.axvline(x=thr, color="k")

        if gmm is not None:
            prob = np.exp(gmm.log_prob(bins[:, None]))
            plt.plot(bins, prob, color="r", linestyle="solid", linewidth=1.5)

        # plt.title(name)
        plt.xlabel("LLR score")
        plt.grid(True)
        # plt.legend()
        plt.savefig(output_file)
        plt.clf()

    @staticmethod
    def _unsup_gmm_calibration(scores):
        """Performs unsupervised calibration on the scores by training a GMM."""
        mask = np.triu(np.ones(scores.shape, dtype=np.bool), 1)
        scores_r = scores[mask].ravel()[:, None]  # N x 1
        gmm_1c = GMM(num_comp=1)
        gmm_1c.fit(scores_r, epochs=1)
        gmm_2c = gmm_1c.split_comp(2)
        e = gmm_2c.fit(scores_r, epochs=20)
        scale = (gmm_2c.mu[0] - gmm_2c.mu[1]) * gmm_2c.Lambda
        bias = (
            (gmm_2c.mu[1] ** 2 - gmm_2c.mu[0] ** 2) * gmm_2c.Lambda / 2
            + np.log(gmm_2c.pi[0])
            - np.log(gmm_2c.pi[1])
        )
        scores = scale * scores + bias
        bic_lambda = 1
        n = len(scores_r)
        dparams = 4
        bic = (
            np.mean(gmm_2c.log_prob(scores_r) - gmm_1c.log_prob(scores_r))
            - bic_lambda * dparams * np.log(n) / 2 / n
        )
        return scores, bic, gmm_2c

    def cluster(self, x, hist_file=None):
        """Peforms the diarization clustering.

        Args:
          x: input data (num_frames, feat_dim)
          hist_file: file to plot the score histogram (optional).

        Returns:
          Cluster assigments as (num_frames,) integer array.
        """
        x = self.preproc.predict(x)
        if self.pca_var_r < 1:
            pca = PCA(pca_var_r=self.pca_var_r, whiten=True)
            pca.fit(x)
            logging.info("PCA dim=%d" % pca.pca_dim)
            x = pca.predict(x)
            x = LNorm().predict(x)
            plda_model = self.plda_model.project(pca.T, pca.mu)
        else:
            plda_model = self.plda_model

        scores = plda_model.llr_1vs1(x, x)
        if self.do_unsup_cal:
            scores_cal, bic, gmm_2c = self._unsup_gmm_calibration(scores)
            logging.info(
                "UnsupCal. BIC={} gmm.pi={} gmm.mu={} gmm.sigma={}".format(
                    bic, gmm_2c.pi, gmm_2c.mu, np.sqrt(1.0 / gmm_2c.Lambda)
                )
            )
            if hist_file:
                hist_file_1 = "%s-nocal.pdf" % hist_file
                self._plot_score_hist(scores, hist_file_1, None, gmm_2c)
                scores = scores_cal

        if hist_file:
            hist_file_1 = "%s.pdf" % hist_file
            self._plot_score_hist(scores, hist_file_1, self.threshold)

        if self.use_bic and bic < 0:
            # unsup calibration detected only one Gaussian -> only target trials
            class_ids = np.zeros(len(x), dtype=np.int)
            return class_ids

        self._ahc.fit(scores)
        class_ids = self._ahc.get_flat_clusters(self.threshold)

        return class_ids

    @staticmethod
    def filter_args(**kwargs):
        """Filters diarization args from arguments dictionary.

        Args:
          prefix: Options prefix.
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with diarization options.
        """
        valid_args = ("threshold", "pca_var_r", "do_unsup_cal", "use_bic")

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds diarization options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """

        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(p1 + "threshold", default=0, type=float)
        parser.add_argument(p1 + "pca-var-r", default=1, type=float)
        parser.add_argument(p1 + "do-unsup-cal", default=False, action="store_true")
        parser.add_argument(p1 + "use-bic", default=False, action="store_true")

    add_argparse_args = add_class_args
