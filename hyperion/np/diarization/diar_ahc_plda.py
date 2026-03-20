"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...utils import PathLike
from ...utils.math_funcs import cosine_scoring
from ...utils.vad_utils import merge_vad_timestamps
from ..clustering import AHC
from ..pdfs import GMMTiedDiagCov as GMM
from ..transforms import PCA


class DiarAHCPLDA:
    """Performs diarization with agglomerative hierarchical clustering (AHC).

    Pipeline:
      1. Optional feature pre-processing (e.g., LDA + length norm).
      2. Optional PCA fit on current utterance and projection of features (and
         PLDA parameters when PLDA is used).
      3. Pairwise scoring with PLDA (or cosine scoring if PLDA is not provided).
      4. Optional score calibration (external calibrator and/or unsupervised GMM).
      5. AHC and optional post-merge of temporal intervals per speaker.

    Attributes:
      plda_model: Pre-trained PLDA-like model. If ``None``, cosine scoring is used.
      preproc: Optional callable transform applied to ``x`` before scoring.
      calibrator: Optional external score calibrator applied element-wise to
        the score matrix.
      threshold: Stopping threshold for AHC flat clustering.
      max_clusters: Optional upper bound on number of output clusters.
      pca_var_r: Variance ratio preserved by PCA in ``(0, 1]``.
        If ``pca_var_r=1``, PCA is skipped.
      do_unsup_cal: If ``True``, runs unsupervised 2-Gaussian score calibration.
      use_bic: If ``True`` (and unsupervised calibration is enabled), uses BIC to
        detect one-Gaussian cases and return a single cluster.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.diarization.diar_ahc_plda import DiarAHCPLDA
      >>> x = np.random.randn(100, 256).astype(np.float32)
      >>> t_start = np.arange(100, dtype=np.float32) * 0.01
      >>> t_end = t_start + 0.01
      >>> diar = DiarAHCPLDA(threshold=0.0, pca_var_r=1.0, do_unsup_cal=False)
      >>> cluster_ids, t_start_out, t_end_out = diar(x, t_start=t_start, t_end=t_end)
    """

    def __init__(
        self,
        plda_model: Optional[Any] = None,
        preproc: Optional[Any] = None,
        calibrator: Optional[Any] = None,
        threshold: float = 0.0,
        max_clusters: Optional[int] = None,
        pca_var_r: float = 1.0,
        do_unsup_cal: bool = False,
        use_bic: bool = False,
    ) -> None:
        """Initializes a diarization backend based on AHC over PLDA scores.

        Args:
          plda_model: Pre-trained PLDA-like model. If ``None``, cosine scoring is used.
          preproc: Optional preprocessing transform/callable applied to features.
          calibrator: Optional external score calibrator.
          threshold: AHC threshold used to cut the dendrogram.
          max_clusters: Optional upper bound on number of output clusters.
          pca_var_r: PCA kept-variance ratio in ``(0, 1]``. ``1`` disables PCA.
          do_unsup_cal: Enables unsupervised GMM score calibration.
          use_bic: Uses BIC decision from unsupervised calibration to force a
            single-cluster output when supported by the data.
        """
        if not (0 < pca_var_r <= 1):
            raise ValueError(f"pca_var_r must be in (0, 1], got {pca_var_r!r}")
        if max_clusters is not None and (
            not isinstance(max_clusters, (int, np.integer)) or max_clusters < 1
        ):
            raise ValueError(
                f"max_clusters must be a positive integer or None, got {max_clusters!r}"
            )

        self.plda_model = plda_model
        self.preproc = preproc
        self.calibrator = calibrator
        self.threshold = threshold
        self.pca_var_r = pca_var_r
        self.do_unsup_cal = do_unsup_cal
        self.use_bic = use_bic and do_unsup_cal
        self.max_clusters = max_clusters
        self._ahc = AHC()

    @staticmethod
    def _plot_score_hist(
        scores: np.ndarray,
        output_file: PathLike,
        thr: Optional[float] = None,
        gmm: Optional[Any] = None,
    ) -> None:
        """Plots score histogram and optional calibration model density.

        Args:
          scores: Pairwise score matrix ``(N, N)``.
          output_file: Output plot path.
          thr: Optional decision threshold to draw as vertical line.
          gmm: Optional fitted GMM object for plotting model density.

        Returns:
          ``None``.
        """
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        mask = np.triu(np.ones(scores.shape, dtype=bool), 1)
        scores_r = scores[mask].ravel()
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Bitstream Vera Sans",
            "Computer Modern Sans Serif",
            "Lucida Grande",
            "Verdana",
            "Geneva",
            "Lucid",
            "Arial",
            "Helvetica",
            "Avant Garde",
            "sans-serif",
        ]
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
    def _unsup_gmm_calibration(scores: np.ndarray) -> Tuple[np.ndarray, float, Any]:
        """Performs unsupervised score calibration using a 2-component GMM.

        Args:
          scores: Pairwise score matrix ``(N, N)``.

        Returns:
          scores_cal: Calibrated scores with same shape as input.
          bic: BIC-based evidence for 2-comp vs 1-comp model.
          gmm_2c: Trained 2-component GMM used for calibration.
        """
        mask = np.triu(np.ones(scores.shape, dtype=bool), 1)
        scores_r = scores[mask].ravel()[:, None]  # N x 1
        gmm_1c = GMM(num_comp=1)
        gmm_1c.fit(scores_r, epochs=1)
        gmm_2c = gmm_1c.split_comp(2)
        gmm_2c.fit(scores_r, epochs=20)
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

    def _merge_intervals(
        self, cluster_ids: np.ndarray, t_start: np.ndarray, t_end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merges overlapping/adjacent intervals per speaker cluster.

        Args:
          cluster_ids: Cluster assignments ``(num_segments,)``.
          t_start: Segment start times ``(num_segments,)``.
          t_end: Segment end times ``(num_segments,)``.

        Returns:
          new_cluster_ids: Reindexed cluster labels sorted by start time.
          new_t_start: Merged segment start times.
          new_t_end: Merged segment end times.
        """
        new_t_start = []
        new_t_end = []
        new_cluster_ids = []
        # print("merge_in", cluster_ids, t_start, t_end, len(cluster_ids))
        for i in np.unique(cluster_ids):
            idx = cluster_ids == i
            t_start_i = t_start[idx]
            t_end_i = t_end[idx]
            t_start_i, t_end_i = merge_vad_timestamps(t_start_i, t_end_i)
            new_t_start.append(t_start_i)
            new_t_end.append(t_end_i)
            new_cluster_ids.append([i] * len(t_start_i))

        new_t_start = np.concatenate(new_t_start)
        new_t_end = np.concatenate(new_t_end)
        new_cluster_ids = np.concatenate(new_cluster_ids)
        # print("merge_out", new_cluster_ids, new_t_start, new_t_end, len(cluster_ids))
        # sort by t start
        idx = np.argsort(new_t_start)
        new_t_start = new_t_start[idx]
        new_t_end = new_t_end[idx]
        new_cluster_ids = new_cluster_ids[idx]
        return new_cluster_ids, new_t_start, new_t_end

    def __call__(
        self,
        x: np.ndarray,
        t_start: Optional[np.ndarray] = None,
        t_end: Optional[np.ndarray] = None,
        hist_file: Optional[PathLike] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Performs diarization clustering.

        Args:
          x: Input feature matrix ``(num_segments, feat_dim)``.
          t_start: Optional segment start times.
          t_end: Optional segment end times.
          hist_file: Optional path to save score histogram plot.

        Returns:
          cluster_ids: Cluster assignments. Shape is ``(num_segments,)`` when no
            interval merge is requested, otherwise ``(num_merged_segments,)``.
          t_start: Input/merged segment start times if provided.
          t_end: Input/merged segment end times if provided.
        """
        if x.ndim != 2:
            raise ValueError(
                f"x must be a 2D array with shape (num_segments, feat_dim), got {x.shape}"
            )
        if self.preproc is not None:
            x = self.preproc(x)

        num_segments = x.shape[0]
        if (t_start is None) != (t_end is None):
            raise ValueError("t_start and t_end must both be provided or both be None")
        if t_start is not None and t_end is not None:
            if len(t_start) != num_segments or len(t_end) != num_segments:
                raise ValueError(
                    "t_start and t_end must have length equal to number of segments: "
                    f"len(t_start)={len(t_start)}, len(t_end)={len(t_end)}, "
                    f"num_segments={num_segments}"
                )
        if num_segments < 2:
            logging.warning(
                "DiarAHCPLDA received %d segment(s); returning trivial clustering",
                num_segments,
            )
            cluster_ids = np.zeros((num_segments,), dtype=int)
            return cluster_ids, t_start, t_end

        if self.pca_var_r < 1:
            pca = PCA(pca_var_r=self.pca_var_r, whiten=True)
            pca.fit(x)
            logging.info("PCA dim=%d" % pca.pca_dim)
            x = pca(x)
            if self.plda_model is None:
                plda_model = None
            else:
                plda_model = self.plda_model.project(pca.T, pca.mu)
        else:
            plda_model = self.plda_model

        if plda_model is None:
            scores = cosine_scoring(x, x)
        else:
            scores = plda_model.llr_1vs1(x, x)

        if self.calibrator is not None:
            scores = self.calibrator(scores.ravel()).reshape(scores.shape)

        if self.do_unsup_cal:
            scores_cal, bic, gmm_2c = self._unsup_gmm_calibration(scores)
            logging.info(
                "UnsupCal. BIC={} gmm.pi={} gmm.mu={} gmm.sigma={}".format(
                    bic, gmm_2c.pi, gmm_2c.mu, np.sqrt(1.0 / gmm_2c.Lambda)
                )
            )
            if hist_file:
                hist_file = Path(hist_file)
                hist_file_1 = hist_file.with_name(
                    f"{hist_file.stem}_nocal{hist_file.suffix}"
                )
                self._plot_score_hist(scores, hist_file_1, None, gmm_2c)

            scores = scores_cal

        if hist_file:
            self._plot_score_hist(scores, hist_file, self.threshold)

        if self.use_bic and bic < 0:
            # unsup calibration detected only one Gaussian -> only target trials
            cluster_ids = np.zeros(len(x), dtype=int)
            return cluster_ids, t_start, t_end

        self._ahc.fit(scores)
        cluster_ids = self._ahc.get_flat_clusters(self.threshold)
        if self.max_clusters is not None and np.max(cluster_ids) >= self.max_clusters:
            cluster_ids = self._ahc.get_flat_clusters(
                self.max_clusters, criterion="num_clusters"
            )
        if t_start is not None and t_end is not None:
            cluster_ids, t_start, t_end = self._merge_intervals(
                cluster_ids, t_start, t_end
            )

        return cluster_ids, t_start, t_end

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters diarization args from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with diarization options.
        """
        valid_args = (
            "threshold",
            "max_clusters",
            "pca_var_r",
            "do_unsup_cal",
            "use_bic",
        )

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return d

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds diarization options to parser.

        Args:
          parser: Arguments parser.
          prefix: Options prefix.

        Returns:
          ``None``.
        """

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--threshold",
            default=0,
            type=float,
            help="AHC threshold used to cut the dendrogram into flat clusters",
        )
        parser.add_argument(
            "--max-clusters",
            default=None,
            type=int,
            help="optional upper bound on number of output clusters",
        )
        parser.add_argument(
            "--pca-var-r",
            default=1,
            type=float,
            help="PCA kept-variance ratio in (0,1]; set to 1 to disable PCA",
        )
        parser.add_argument(
            "--do-unsup-cal",
            default=False,
            action=ActionYesNo,
            help="enable unsupervised GMM-based score calibration",
        )
        parser.add_argument(
            "--use-bic",
            default=False,
            action=ActionYesNo,
            help="with unsupervised calibration, use BIC to force one-cluster output when appropriate",
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
