"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from scipy.special import ndtri

from ...utils.sparse_trial_key import SparseTrialKey
from ...utils.sparse_trial_scores import SparseTrialScores
from ...utils.trial_key import TrialKey
from ...utils.trial_scores import TrialScores
from ...utils.misc import PathLike
from .dcf import compute_act_dcf
from .roc import compute_roc, compute_rocch


class DETPlotWindowType(str, Enum):
    """Predefined DET window presets."""

    SRE12 = "sre12"
    SRE10 = "sre10"
    OLD = "old"
    BIG = "big"

    @staticmethod
    def choices() -> List["DETPlotWindowType"]:
        """Returns valid preset choices.

        Returns:
            List[DETPlotWindowType]: Supported window preset enum values.
        """
        return [e for e in DETPlotWindowType]


@dataclass
class DETPlotWindow:
    """DET axis limits/ticks/labels configuration.

    Attributes:
        pfa_limits: Two-element vector with min/max false-alarm probability.
        pmiss_limits: Two-element vector with min/max miss probability.
        xticks: False-alarm tick positions (probability domain).
        xtick_labels: Labels corresponding to ``xticks``.
        yticks: Miss-probability tick positions (probability domain).
        ytick_labels: Labels corresponding to ``yticks``.
    """

    pfa_limits: np.ndarray
    pmiss_limits: np.ndarray
    xticks: np.ndarray
    xtick_labels: Sequence[str]
    yticks: np.ndarray
    ytick_labels: Sequence[str]

    def __post_init__(self) -> None:
        """Normalizes dtypes/labels and validates field consistency."""
        self.pfa_limits = np.asarray(self.pfa_limits, dtype=float)
        self.pmiss_limits = np.asarray(self.pmiss_limits, dtype=float)
        self.xticks = np.asarray(self.xticks, dtype=float)
        self.yticks = np.asarray(self.yticks, dtype=float)
        self.xtick_labels = [str(x).strip() for x in self.xtick_labels]
        self.ytick_labels = [str(y).strip() for y in self.ytick_labels]

        if self.pfa_limits.shape != (2,):
            raise ValueError("pfa_limits must have shape (2,)")
        if self.pmiss_limits.shape != (2,):
            raise ValueError("pmiss_limits must have shape (2,)")
        if len(self.xticks) != len(self.xtick_labels):
            raise ValueError("xticks and xtick_labels must have the same length")
        if len(self.yticks) != len(self.ytick_labels):
            raise ValueError("yticks and ytick_labels must have the same length")

    @staticmethod
    def _axis_sre12() -> "DETPlotWindow":
        """Builds SRE12-like default DET window.

        Returns:
            DETPlotWindow: Preset window configuration.
        """
        return DETPlotWindow(
            pfa_limits=np.array([5e-6, 5e-3], dtype=float),
            pmiss_limits=np.array([1e-2, 0.99], dtype=float),
            xticks=np.array(
                [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3], dtype=float
            ),
            xtick_labels=["1e-3", "2e-3", "5e-3", "0.01", "0.02", "0.05", "0.1", "0.2"],
            yticks=np.array(
                [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
                dtype=float,
            ),
            ytick_labels=[
                "2",
                "5",
                "10",
                "20",
                "30",
                "40",
                "50",
                "60",
                "70",
                "80",
                "90",
                "95",
                "98",
            ],
        )

    @staticmethod
    def _axis_old() -> "DETPlotWindow":
        """Builds legacy ("old") DET window.

        Returns:
            DETPlotWindow: Preset window configuration.
        """
        ticks = np.array(
            [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4], dtype=float
        )
        labels = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "30", "40"]
        return DETPlotWindow(
            pfa_limits=np.array([5e-4, 5e-1], dtype=float),
            pmiss_limits=np.array([5e-4, 5e-1], dtype=float),
            xticks=ticks,
            xtick_labels=labels,
            yticks=ticks,
            ytick_labels=labels,
        )

    @staticmethod
    def _axis_big() -> "DETPlotWindow":
        """Builds large-range ("big") DET window.

        Returns:
            DETPlotWindow: Preset window configuration.
        """
        yticks = np.array(
            [
                5e-6,
                5e-5,
                5e-4,
                0.5e-2,
                2.5e-2,
                10e-2,
                25e-2,
                50e-2,
                72e-2,
                88e-2,
                96e-2,
                99e-2,
            ],
            dtype=float,
        )
        ytick_labels = [
            "5e-4",
            "5e-3",
            "0.05",
            "0.5",
            "2.5",
            "10",
            "25",
            "50",
            "72",
            "88",
            "96",
            "99",
        ]
        return DETPlotWindow(
            pfa_limits=np.array([5e-6, 0.99], dtype=float),
            pmiss_limits=np.array([5e-6, 0.99], dtype=float),
            xticks=yticks[1:],
            xtick_labels=ytick_labels[1:],
            yticks=yticks,
            ytick_labels=ytick_labels,
        )

    @staticmethod
    def _axis_sre10() -> "DETPlotWindow":
        """Builds SRE10-like DET window.

        Returns:
            DETPlotWindow: Preset window configuration.
        """
        return DETPlotWindow(
            pfa_limits=np.array([3e-6, 5e-1], dtype=float),
            pmiss_limits=np.array([3e-4, 9e-1], dtype=float),
            xticks=np.array(
                [1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1],
                dtype=float,
            ),
            xtick_labels=[
                "0.001",
                "0.01",
                "0.1",
                "0.2",
                "0.5",
                "1",
                "2",
                "5",
                "10",
                "20",
                "40",
            ],
            yticks=np.array(
                [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1, 8e-1],
                dtype=float,
            ),
            ytick_labels=["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40", "80"],
        )

    @classmethod
    def make_window(
        cls, window_type: Union["DETPlotWindowType", str]
    ) -> "DETPlotWindow":
        """Builds a :class:`DETPlotWindow` from a named preset.

        Args:
            window_type: Window preset enum or its string value.

        Returns:
            DETPlotWindow: Preset window instance.
        """
        if isinstance(window_type, str):
            window_type = DETPlotWindowType(window_type)

        if window_type == DETPlotWindowType.SRE12:
            return cls._axis_sre12()
        if window_type == DETPlotWindowType.SRE10:
            return cls._axis_sre10()
        if window_type == DETPlotWindowType.OLD:
            return cls._axis_old()
        if window_type == DETPlotWindowType.BIG:
            return cls._axis_big()
        raise ValueError(f"Unsupported DET plot window type '{window_type}'")

    @classmethod
    def make_det_plot_window(
        cls, window_type: Union["DETPlotWindowType", str]
    ) -> "DETPlotWindow":
        """Backward-compatible alias for :meth:`make_window`.

        Args:
            window_type: Window preset enum or its string value.

        Returns:
            DETPlotWindow: Preset window instance.
        """
        return cls.make_window(window_type)


DETPlotWindowInput = Union[DETPlotWindow, DETPlotWindowType, str]


class DETPlot:
    """DET plotting helper with BOSARIS-like API.

    Attributes:
        plot_window: Active DET axis configuration.
        plot_title: Optional figure title.
        priors: Sorted target priors used for min/act DCF points.
        fh: Matplotlib figure handle (created lazily).
        ax: Matplotlib axes handle (created lazily).
        handles_vec: Plotted handles included in the legend.
        legend_strings: Legend strings aligned with ``handles_vec``.

    Examples:
        >>> det = DETPlot("sre10", plot_title="DET", priors=[0.01, 0.1])
        >>> h = det.plot_curve_from_scores(
        ...     tar_scores, non_scores, method="rocch", system_name="sysA", color="b", line_type="-"
        ... )
        >>> det.plot_dr30(num_tar=len(tar_scores), num_non=len(non_scores))
        >>> det.save("det.png", dpi=200)
    """

    def __init__(
        self,
        plot_window: DETPlotWindowInput,
        plot_title: Optional[str] = None,
        priors: Union[float, np.ndarray, Sequence[float]] = 0.5,
    ) -> None:
        """Initializes DET plotting state.

        Args:
            plot_window: Window object or named preset.
            plot_title: Optional title for the figure.
            priors: One or many target priors in ``(0, 1)``.
        """
        if isinstance(plot_window, DETPlotWindow):
            plot_window_obj = plot_window
        else:
            plot_window_obj = DETPlotWindow.make_window(plot_window)

        self.plot_window = plot_window_obj

        self.plot_title = plot_title

        priors = np.asarray(priors, dtype=float)
        if priors.ndim == 0:
            priors = priors[None]
        if np.any((priors <= 0) | (priors >= 1)):
            raise ValueError("priors must be strictly between 0 and 1")
        self.priors = np.sort(priors, kind="mergesort")

        self.fh = None
        self.ax = None
        self.handles_vec = []
        self.legend_strings = []
        self.tar = None
        self.non = None
        self.sys_name = None

    @staticmethod
    def _probit(p: np.ndarray) -> np.ndarray:
        """Maps probabilities to normal-deviate (probit) axis.

        Args:
            p: Probability values.

        Returns:
            np.ndarray: Probit-transformed values.
        """
        eps = np.finfo(float).eps
        p = np.asarray(p, dtype=float)
        # ndtri is the inverse CDF (quantile function) of the standard normal distribution.
        # Input: probability p in (0, 1)
        # Output: z such that Phi(z) = p (where Phi is normal CDF)
        # In DET plots, it converts probabilities (pfa, pmiss) to normal-deviate coordinates so Gaussian-like error tradeoffs look more linear.
        return ndtri(np.clip(p, eps, 1.0 - eps))

    def _ensure_axes(self) -> Any:
        """Creates and configures the DET axes if needed.

        Returns:
            Any: Matplotlib axes object.
        """
        if self.ax is not None:
            return self.ax

        import matplotlib.pyplot as plt

        self.fh, self.ax = plt.subplots()
        self.ax.set_xlim(self._probit(self.pfa_limits))
        self.ax.set_xticks(self._probit(self.xticks))
        self.ax.set_xticklabels(self.xtick_labels)
        self.ax.set_xlabel("False Alarm probability (in %)")
        self.ax.grid(True, axis="x")

        self.ax.set_ylim(self._probit(self.pmiss_limits))
        self.ax.set_yticks(self._probit(self.yticks))
        self.ax.set_yticklabels(self.ytick_labels)
        self.ax.set_ylabel("Miss probability (in %)")
        self.ax.grid(True, axis="y")

        self.ax.set_aspect("equal", adjustable="box")
        if self.plot_title:
            self.ax.set_title(self.plot_title)
        return self.ax

    def plot_curve_from_roc(
        self,
        pmiss: np.ndarray,
        pfa: np.ndarray,
        system_name: str,
        color: str,
        line_type: str,
        line_width: float = 1.5,
        min_dcf: bool = False,
    ) -> Any:
        """Plots a DET curve from precomputed ROC points.

        Args:
            pmiss: Miss-probability samples.
            pfa: False-alarm probability samples.
            system_name: Legend label for the curve.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            line_width: Matplotlib line width for the DET curve.
            min_dcf: If True, marks min-DCF points for each prior.

        Returns:
            Any: Matplotlib line handle for the main curve.
        """
        pmiss = np.asarray(pmiss, dtype=float).ravel()
        pfa = np.asarray(pfa, dtype=float).ravel()
        if pmiss.shape != pfa.shape:
            raise ValueError("pmiss and pfa must have the same shape")
        if pmiss.size == 0:
            raise ValueError("pmiss and pfa cannot be empty")

        pmiss_all = pmiss
        pfa_all = pfa

        in_window = (
            (pfa >= self.pfa_limits[0])
            & (pfa <= self.pfa_limits[1])
            & (pmiss >= self.pmiss_limits[0])
            & (pmiss <= self.pmiss_limits[1])
        )
        if np.any(in_window):
            pfa = pfa[in_window]
            pmiss = pmiss[in_window]

        ax = self._ensure_axes()
        (line_handle,) = ax.plot(
            self._probit(pfa),
            self._probit(pmiss),
            color=color,
            linestyle=line_type,
            linewidth=line_width,
            label=system_name,
        )

        if min_dcf:
            for prior in self.priors:
                dcf = prior * pmiss_all + (1.0 - prior) * pfa_all
                idx_min = int(np.argmin(dcf))
                ax.plot(
                    self._probit(pfa_all[idx_min]),
                    self._probit(pmiss_all[idx_min]),
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=1.5,
                    color=color,
                    label="_nolegend_",
                )

        if system_name:
            self.handles_vec.append(line_handle)
            self.legend_strings.append(system_name)
            ax.legend(self.handles_vec, self.legend_strings)

        return line_handle

    def plot_curve_from_scores(
        self,
        tar_scores: np.ndarray,
        non_scores: np.ndarray,
        method: str = "rocch",
        system_name: str = "",
        color: str = "b",
        line_type: str = "-",
        line_width: float = 1.5,
        min_dcf: bool = False,
        act_dcf: bool = False,
    ) -> Any:
        """Computes ROC/ROCCH from scores and plots the DET curve.

        Args:
            tar_scores: Target-trial scores.
            non_scores: Non-target-trial scores.
            method: ``"rocch"`` or ``"steppy"``.
            system_name: Legend label for the curve.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            line_width: Matplotlib line width for the DET curve.
            min_dcf: If True, marks min-DCF points for each prior.
            act_dcf: If True, plots actual-DCF operating points with confidence crosses.

        Returns:
            Any: Matplotlib line handle for the main curve.
        """
        tar_scores = np.asarray(tar_scores, dtype=float).ravel()
        non_scores = np.asarray(non_scores, dtype=float).ravel()
        if tar_scores.size == 0 or non_scores.size == 0:
            raise ValueError("tar_scores and non_scores cannot be empty")

        method = method.lower()
        if method == "steppy":
            pmiss, pfa = compute_roc(tar_scores, non_scores)
        elif method == "rocch":
            pmiss, pfa = compute_rocch(tar_scores, non_scores)
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Valid options are 'rocch' and 'steppy'"
            )

        line_handle = self.plot_curve_from_roc(
            pmiss=pmiss,
            pfa=pfa,
            system_name=system_name,
            color=color,
            line_type=line_type,
            line_width=line_width,
            min_dcf=min_dcf,
        )

        if act_dcf:
            _, act_pmiss, act_pfa = compute_act_dcf(
                tar_scores, non_scores, prior=self.priors
            )
            act_pmiss = np.atleast_1d(act_pmiss)
            act_pfa = np.atleast_1d(act_pfa)
            ntar = tar_scores.size
            nnon = non_scores.size
            for pm_i, pf_i in zip(act_pmiss, act_pfa):
                self.plot_dcf_point(
                    pmiss=float(pm_i),
                    pfa=float(pf_i),
                    ntar=ntar,
                    nnon=nnon,
                    color=color,
                )

        return line_handle

    def plot_curve_from_trials(
        self,
        key: Union[TrialKey, SparseTrialKey],
        scores: Union[TrialScores, SparseTrialScores],
        method: str = "rocch",
        system_name: str = "",
        color: str = "b",
        line_type: str = "-",
        line_width: float = 1.5,
        min_dcf: bool = False,
        act_dcf: bool = False,
    ) -> Any:
        """Extracts target/non-target scores from trial objects and plots DET.

        Args:
            key: Trial key object with target/non-target masks.
            scores: Trial scores object aligned to the trial space.
            method: ``"rocch"`` or ``"steppy"``.
            system_name: Legend label for the curve.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            line_width: Matplotlib line width for the DET curve.
            min_dcf: If True, marks min-DCF points for each prior.
            act_dcf: If True, plots actual-DCF operating points with confidence crosses.

        Returns:
            Any: Matplotlib line handle for the main curve.
        """
        tar_scores, non_scores = scores.get_tar_non(key)
        return self.plot_curve_from_scores(
            tar_scores=tar_scores,
            non_scores=non_scores,
            method=method,
            system_name=system_name,
            color=color,
            line_type=line_type,
            line_width=line_width,
            min_dcf=min_dcf,
            act_dcf=act_dcf,
        )

    def plot_dcf_point(
        self,
        pmiss: float,
        pfa: float,
        ntar: Optional[int] = None,
        nnon: Optional[int] = None,
        color: Optional[str] = None,
        marker: str = "+",
    ) -> Any:
        """Plots a DCF operating point marker or confidence cross.

        Args:
            pmiss: Miss probability at the operating point.
            pfa: False-alarm probability at the operating point.
            ntar: Number of target trials (required with ``nnon`` for CI cross).
            nnon: Number of non-target trials (required with ``ntar`` for CI cross).
            color: Marker/line color. If None, uses the last curve color.
            marker: Marker style used when CI is not requested.

        Returns:
            Any: Matplotlib handle for the plotted artist, or ``None`` if skipped.
        """
        pmiss = float(pmiss)
        pfa = float(pfa)
        if not (0.0 < pmiss < 1.0 and 0.0 < pfa < 1.0):
            raise ValueError("pmiss and pfa must be strictly between 0 and 1")

        if (pfa < self.pfa_limits[0]) or (pfa > self.pfa_limits[1]):
            warnings.warn(
                f"pfa of {pfa:.6f} is not between {self.pfa_limits[0]:.6f} and "
                f"{self.pfa_limits[1]:.6f}. The DCF point will not be plotted.",
                stacklevel=2,
            )
            return None
        if (pmiss < self.pmiss_limits[0]) or (pmiss > self.pmiss_limits[1]):
            warnings.warn(
                f"pmiss of {pmiss:.6f} is not between {self.pmiss_limits[0]:.6f} and "
                f"{self.pmiss_limits[1]:.6f}. The DCF point will not be plotted.",
                stacklevel=2,
            )
            return None

        ax = self._ensure_axes()
        if color is None:
            if len(self.handles_vec) > 0 and hasattr(self.handles_vec[-1], "get_color"):
                color = self.handles_vec[-1].get_color()
            elif len(ax.lines) > 0 and hasattr(ax.lines[-1], "get_color"):
                color = ax.lines[-1].get_color()
            else:
                color = "k"

        if (ntar is None) != (nnon is None):
            raise ValueError("ntar and nnon must be both None or both provided")

        if ntar is None and nnon is None:
            (h,) = ax.plot(
                self._probit(pfa),
                self._probit(pmiss),
                color=color,
                marker=marker,
                linestyle="None",
                markersize=10,
                markeredgewidth=2.0,
                label="_nolegend_",
            )
            return h

        ntar = int(ntar)
        nnon = int(nnon)
        if ntar <= 0 or nnon <= 0:
            raise ValueError("ntar and nnon must be > 0")

        confidence = 0.95
        fctr = float(ndtri(1.0 - (1.0 - confidence) * 0.5))
        stdm = fctr * np.sqrt(pmiss * (1.0 - pmiss) / ntar)
        stdf = fctr * np.sqrt(pfa * (1.0 - pfa) / nnon)
        pml = pmiss - stdm
        pmh = pmiss + stdm
        pfl = pfa - stdf
        pfh = pfa + stdf

        (h_h,) = ax.plot(
            self._probit(np.array([pfl, pfh], dtype=float)),
            self._probit(np.array([pmiss, pmiss], dtype=float)),
            color=color,
            linestyle="-",
            linewidth=2.0,
            label="_nolegend_",
        )
        ax.plot(
            self._probit(np.array([pfa, pfa], dtype=float)),
            self._probit(np.array([pml, pmh], dtype=float)),
            color=color,
            linestyle="-",
            linewidth=2.0,
            label="_nolegend_",
        )
        return h_h

    def save(self, img_path: PathLike, dpi: Optional[int] = None, **kwargs: Any) -> None:
        """Saves the DET figure to an image file.

        Args:
            img_path: Output image path (e.g., ``.png``, ``.pdf``).
            dpi: Optional output DPI.
            **kwargs: Extra keyword args forwarded to ``savefig``.
        """
        self._ensure_axes()
        save_kwargs = dict(kwargs)
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        self.fh.savefig(str(img_path), **save_kwargs)

    def plot_dr30_fa(
        self,
        num_non: int,
        color: str = "k",
        line_type: str = "--",
        line_width: float = 1.5,
        legend_string: str = "",
    ) -> Any:
        """Plots Doddington Rule-of-30 false-alarm vertical line.

        Args:
            num_non: Number of non-target trials.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            line_width: Matplotlib line width for the DR30 line.
            legend_string: Optional legend label for this line.

        Returns:
            Any: Matplotlib line handle, or ``None`` if outside plot limits.
        """
        num_non = int(num_non)
        if num_non <= 0:
            raise ValueError("num_non must be > 0")

        pfaval = 30.0 / num_non
        pfa_min, pfa_max = self.pfa_limits
        if (pfaval < pfa_min) or (pfaval > pfa_max):
            warnings.warn(
                f"Pfa DR30 of {pfaval:.6f} is not between {pfa_min:.6f} and "
                f"{pfa_max:.6f}. Pfa DR30 line will not be plotted.",
                stacklevel=2,
            )
            return None

        ax = self._ensure_axes()
        pmiss_min, pmiss_max = self.pmiss_limits
        x = self._probit(np.array([pfaval, pfaval], dtype=float))
        y = self._probit(np.array([pmiss_min, pmiss_max], dtype=float))
        (line_handle,) = ax.plot(
            x,
            y,
            color=color,
            linestyle=line_type,
            linewidth=line_width,
            label=legend_string if legend_string else "_nolegend_",
        )

        if legend_string:
            self.handles_vec.append(line_handle)
            self.legend_strings.append(legend_string)
            ax.legend(self.handles_vec, self.legend_strings)

        return line_handle

    def plot_dr30_pmiss(
        self,
        num_tar: int,
        color: str = "k",
        line_type: str = "--",
        line_width: float = 1.5,
        legend_string: str = "",
    ) -> Any:
        """Plots Doddington Rule-of-30 miss-rate horizontal line.

        Args:
            num_tar: Number of target trials.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            line_width: Matplotlib line width for the DR30 line.
            legend_string: Optional legend label for this line.

        Returns:
            Any: Matplotlib line handle, or ``None`` if outside plot limits.
        """
        num_tar = int(num_tar)
        if num_tar <= 0:
            raise ValueError("num_tar must be > 0")

        pmissval = 30.0 / num_tar
        pmiss_min, pmiss_max = self.pmiss_limits
        if (pmissval < pmiss_min) or (pmissval > pmiss_max):
            warnings.warn(
                f"Pmiss DR30 of {pmissval:.6f} is not between {pmiss_min:.6f} and "
                f"{pmiss_max:.6f}. Pmiss DR30 line will not be plotted.",
                stacklevel=2,
            )
            return None

        ax = self._ensure_axes()
        pfa_min, pfa_max = self.pfa_limits
        x = self._probit(np.array([pfa_min, pfa_max], dtype=float))
        y = self._probit(np.array([pmissval, pmissval], dtype=float))
        (line_handle,) = ax.plot(
            x,
            y,
            color=color,
            linestyle=line_type,
            linewidth=line_width,
            label=legend_string if legend_string else "_nolegend_",
        )

        if legend_string:
            self.handles_vec.append(line_handle)
            self.legend_strings.append(legend_string)
            ax.legend(self.handles_vec, self.legend_strings)

        return line_handle

    def plot_dr30(
        self,
        num_tar: int,
        num_non: int,
        color: str = "k",
        line_type: str = "--",
        line_width: float = 1.5,
        legend_fa: str = "",
        legend_pmiss: str = "",
    ) -> Any:
        """Plots both DR30 lines (false alarm and miss rate).

        Args:
            num_tar: Number of target trials.
            num_non: Number of non-target trials.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            line_width: Matplotlib line width for both DR30 lines.
            legend_fa: Optional legend label for the false-alarm line.
            legend_pmiss: Optional legend label for the miss-rate line.

        Returns:
            Any: Tuple with handles ``(h_fa, h_pmiss)``.
        """
        h_fa = self.plot_dr30_fa(
            num_non=num_non,
            color=color,
            line_type=line_type,
            line_width=line_width,
            legend_string=legend_fa,
        )
        h_pmiss = self.plot_dr30_pmiss(
            num_tar=num_tar,
            color=color,
            line_type=line_type,
            line_width=line_width,
            legend_string=legend_pmiss,
        )
        return h_fa, h_pmiss

    @property
    def pfa_limits(self) -> np.ndarray:
        return self.plot_window.pfa_limits

    @pfa_limits.setter
    def pfa_limits(self, value: np.ndarray) -> None:
        self.plot_window.pfa_limits = np.asarray(value, dtype=float)

    @property
    def pmiss_limits(self) -> np.ndarray:
        return self.plot_window.pmiss_limits

    @pmiss_limits.setter
    def pmiss_limits(self, value: np.ndarray) -> None:
        self.plot_window.pmiss_limits = np.asarray(value, dtype=float)

    @property
    def xticks(self) -> np.ndarray:
        return self.plot_window.xticks

    @xticks.setter
    def xticks(self, value: np.ndarray) -> None:
        self.plot_window.xticks = np.asarray(value, dtype=float)

    @property
    def xtick_labels(self) -> List[str]:
        return self.plot_window.xtick_labels

    @xtick_labels.setter
    def xtick_labels(self, value: Sequence[str]) -> None:
        self.plot_window.xtick_labels = [str(x).strip() for x in value]

    @property
    def yticks(self) -> np.ndarray:
        return self.plot_window.yticks

    @yticks.setter
    def yticks(self, value: np.ndarray) -> None:
        self.plot_window.yticks = np.asarray(value, dtype=float)

    @property
    def ytick_labels(self) -> List[str]:
        return self.plot_window.ytick_labels

    @ytick_labels.setter
    def ytick_labels(self, value: Sequence[str]) -> None:
        self.plot_window.ytick_labels = [str(y).strip() for y in value]

    @property
    def xticklabels(self) -> Sequence[str]:
        """Backward-compatible alias for xtick_labels."""
        return self.xtick_labels

    @xticklabels.setter
    def xticklabels(self, value: Sequence[str]) -> None:
        self.xtick_labels = list(value)

    @property
    def yticklabels(self) -> Sequence[str]:
        """Backward-compatible alias for ytick_labels."""
        return self.ytick_labels

    @yticklabels.setter
    def yticklabels(self, value: Sequence[str]) -> None:
        self.ytick_labels = list(value)
