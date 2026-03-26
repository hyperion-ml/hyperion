"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, List, Optional, Union

import numpy as np
from scipy.special import expit, logit
import warnings

from ...utils.sparse_trial_key import SparseTrialKey
from ...utils.sparse_trial_scores import SparseTrialScores
from ...utils.trial_key import TrialKey
from ...utils.trial_scores import TrialScores
from .dcf import compute_act_dcf, compute_min_dcf


class NormDCFPlot:
    """Normalized DCF plot scaffold based on BOSARIS ``Norm_DCF_Plot``.

    Attributes:
        fh: Matplotlib figure handle.
        ax: Matplotlib axes handle.
        plot_axes: Plot limits ``[xmin, xmax, ymin, ymax]`` where x is ``logit(Ptar)``.
        plo: Prior-logit grid used to evaluate DCF curves.
        Ptar_norm: Normalization multiplier for miss component.
        Pnon_norm: Normalization multiplier for false-alarm component.
        sys_name: Current system name used in default legends.
        color: Current default color for the active system.
        actDCF: Actual normalized DCF curve for current system.
        actPmiss: Actual miss probabilities for current system.
        actPfa: Actual false-alarm probabilities for current system.
        minDCF: Minimum normalized DCF curve for current system.
        minPmiss: Miss probabilities at minimum DCF.
        minPfa: False-alarm probabilities at minimum DCF.
        dr30Miss: Index on ``plo`` for DR30 miss point.
        dr30FA: Index on ``plo`` for DR30 false-alarm point.
        handles_vec: Plotted handles included in the legend.
        legend_strings: Legend strings aligned with ``handles_vec``.

    Examples:
        >>> plot = NormDCFPlot(min_prior=1e-3, max_prior=0.5, plot_title="Norm DCF")
        >>> plot.set_system_from_scores(tar_scores, non_scores, system_name="sysA", color="b")
        >>> plot.plot_both_dcf()
        >>> plot.plot_dr30()
        >>> plot.save("norm_dcf.png", dpi=200)
    """

    def __init__(
        self,
        min_prior: float = 0.001,
        max_prior: float = 0.5,
        min_dcf: float = 0.0,
        max_dcf: float = 1.2,
        plot_title: Optional[str] = None,
    ) -> None:
        """Initializes a normalized DCF plot.

        Args:
            min_prior: Minimum effective target prior shown on the x-axis.
            max_prior: Maximum effective target prior shown on the x-axis.
            min_dcf: Minimum normalized DCF shown on the y-axis.
            max_dcf: Maximum normalized DCF shown on the y-axis.
            plot_title: Optional figure title.
        """
        if not (0.0 < min_prior < max_prior < 1.0):
            raise ValueError("priors must satisfy 0 < min_prior < max_prior < 1")
        if min_dcf >= max_dcf:
            raise ValueError("min_dcf must be < max_dcf")

        xmin = float(logit(min_prior))
        xmax = float(logit(max_prior))
        self.plot_axes = np.array([xmin, xmax, min_dcf, max_dcf], dtype=float)

        self.plo = np.linspace(xmin, xmax, 1001, dtype=float)
        p_tar = expit(self.plo)
        p_non = expit(-self.plo)
        ref_pe = np.minimum(p_tar, p_non)
        self.Ptar_norm = p_tar / ref_pe
        self.Pnon_norm = p_non / ref_pe

        self.sys_name = None
        self.color = None
        self.actDCF = None
        self.actPmiss = None
        self.actPfa = None
        self.minDCF = None
        self.minPmiss = None
        self.minPfa = None
        self.dr30Miss = None
        self.dr30FA = None

        self.handles_vec: List[Any] = []
        self.legend_strings: List[str] = []

        import matplotlib.pyplot as plt

        self.fh, self.ax = plt.subplots()
        self.ax.set_ylabel("normalized DCF")
        self.ax.set_xlabel(r"logit $P_{tar}$")
        self.ax.grid(True)
        self.ax.axis(self.plot_axes)
        if plot_title:
            self.ax.set_title(plot_title)

    def _resolve_color(self, color: Optional[str]) -> str:
        """Resolves plotting color from method input and current system state."""
        if color is not None:
            return color
        if self.color is not None:
            return self.color
        return "k"

    def _resolve_legend(self, legend: Optional[str], prefix: str) -> str:
        """Resolves legend text using optional user text and system name."""
        if legend is not None:
            return legend
        if self.sys_name:
            return f"{prefix} {self.sys_name}"
        return prefix

    def _plot_series(
        self,
        y: np.ndarray,
        color: Optional[str],
        line_type: Any,
        legend: Optional[str],
    ) -> Any:
        """Plots a y-series on ``self.plo`` and optionally registers legend entry.

        Args:
            y: Y values to plot against ``self.plo``.
            color: Optional line color.
            line_type: Matplotlib line style.
            legend: Optional legend text.

        Returns:
            Any: Matplotlib line handle.
        """
        c = self._resolve_color(color)
        (h,) = self.ax.plot(
            self.plo,
            y,
            color=c,
            linestyle=line_type,
            label=legend if legend else "_nolegend_",
        )
        if legend:
            self.handles_vec.append(h)
            self.legend_strings.append(legend)
            self.ax.legend(self.handles_vec, self.legend_strings)
        return h

    def set_system_from_scores(
        self,
        tar_scores: np.ndarray,
        non_scores: np.ndarray,
        system_name: str = "",
        color: Optional[str] = None,
    ) -> None:
        """Sets current system from target/non-target score arrays.

        Args:
            tar_scores: Target-trial scores.
            non_scores: Non-target-trial scores.
            system_name: Optional system name to prefix curve labels.
            color: Optional default color for this system.
        """
        tar_scores = np.asarray(tar_scores, dtype=float).ravel()
        non_scores = np.asarray(non_scores, dtype=float).ravel()
        if tar_scores.size == 0 or non_scores.size == 0:
            raise ValueError("tar_scores and non_scores cannot be empty")

        self.sys_name = system_name if system_name is not None else ""
        self.color = color

        priors = expit(self.plo)
        self.actDCF, self.actPmiss, self.actPfa = compute_act_dcf(
            tar_scores, non_scores, priors, normalize=True
        )
        self.minDCF, self.minPmiss, self.minPfa = compute_min_dcf(
            tar_scores, non_scores, priors, normalize=True
        )

        pfa30 = 30.0 / non_scores.size
        idx_fa = np.flatnonzero(self.minPfa >= pfa30)
        self.dr30FA = int(idx_fa[0]) if idx_fa.size > 0 else None

        pmiss30 = 30.0 / tar_scores.size
        idx_miss = np.flatnonzero(self.minPmiss >= pmiss30)
        self.dr30Miss = int(idx_miss[-1]) if idx_miss.size > 0 else None

    def set_system_from_trials(
        self,
        key: Union[TrialKey, SparseTrialKey],
        scores: Union[TrialScores, SparseTrialScores],
        system_name: str = "",
        color: Optional[str] = None,
    ) -> None:
        """Sets current system from trial objects.

        Args:
            key: Trial key with target/non-target trial masks.
            scores: Trial scores container.
            system_name: Optional system name to prefix curve labels.
            color: Optional default color for this system.
        """
        tar_scores, non_scores = scores.get_tar_non(key)
        self.set_system_from_scores(
            tar_scores=tar_scores,
            non_scores=non_scores,
            system_name=system_name,
            color=color,
        )

    def plot_min_dcf(
        self,
        color: Optional[str] = None,
        line_type: Any = "-",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots minimum normalized DCF curve.

        Args:
            color: Optional curve color.
            line_type: Matplotlib line style.
            legend: Optional legend text. Defaults to ``"MinDCF {system_name}"``.

        Returns:
            Any: Matplotlib line handle.
        """
        if self.minDCF is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        legend = self._resolve_legend(legend, "MinDCF")
        return self._plot_series(self.minDCF, color, line_type, legend)

    def plot_act_dcf(
        self,
        color: Optional[str] = None,
        line_type: Any = "--",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots actual normalized DCF curve.

        Args:
            color: Optional curve color.
            line_type: Matplotlib line style.
            legend: Optional legend text. Defaults to ``"ActDCF {system_name}"``.

        Returns:
            Any: Matplotlib line handle.
        """
        if self.actDCF is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        legend = self._resolve_legend(legend, "ActDCF")
        return self._plot_series(self.actDCF, color, line_type, legend)

    def plot_mindcf_pmiss(
        self,
        color: Optional[str] = None,
        line_type: Any = "-.",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots miss contribution to minimum normalized DCF.

        Args:
            color: Optional curve color.
            line_type: Matplotlib line style.
            legend: Optional legend text. Defaults to ``"MinDCF PMiss {system_name}"``.

        Returns:
            Any: Matplotlib line handle.
        """
        if self.minPmiss is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        legend = self._resolve_legend(legend, "MinDCF PMiss")
        y = self.minPmiss * self.Ptar_norm
        return self._plot_series(y, color, line_type, legend)

    def plot_mindcf_pfa(
        self,
        color: Optional[str] = None,
        line_type: Any = ":",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots false-alarm contribution to minimum normalized DCF.

        Args:
            color: Optional curve color.
            line_type: Matplotlib line style.
            legend: Optional legend text. Defaults to ``"MinDCF PFA {system_name}"``.

        Returns:
            Any: Matplotlib line handle.
        """
        if self.minPfa is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        legend = self._resolve_legend(legend, "MinDCF PFA")
        y = self.minPfa * self.Pnon_norm
        return self._plot_series(y, color, line_type, legend)

    def plot_actdcf_pmiss(
        self,
        color: Optional[str] = None,
        line_type: Any = (0, (1, 1)),
        legend: Optional[str] = None,
    ) -> Any:
        """Plots miss contribution to actual normalized DCF.

        Args:
            color: Optional curve color.
            line_type: Matplotlib line style.
            legend: Optional legend text. Defaults to ``"ActDCF PMiss {system_name}"``.

        Returns:
            Any: Matplotlib line handle.
        """
        if self.actPmiss is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        legend = self._resolve_legend(legend, "ActDCF PMiss")
        y = self.actPmiss * self.Ptar_norm
        return self._plot_series(y, color, line_type, legend)

    def plot_actdcf_pfa(
        self,
        color: Optional[str] = None,
        line_type: Any = (0, (3, 1, 1, 1)),
        legend: Optional[str] = None,
    ) -> Any:
        """Plots false-alarm contribution to actual normalized DCF.

        Args:
            color: Optional curve color.
            line_type: Matplotlib line style.
            legend: Optional legend text. Defaults to ``"ActDCF PFA {system_name}"``.

        Returns:
            Any: Matplotlib line handle.
        """
        if self.actPfa is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        legend = self._resolve_legend(legend, "ActDCF PFA")
        y = self.actPfa * self.Pnon_norm
        return self._plot_series(y, color, line_type, legend)

    def plot_both_dcf(self, color: Optional[str] = None) -> Any:
        """Plots both minimum and actual DCF curves.

        Args:
            color: Optional shared color for both curves.

        Returns:
            Any: Tuple with handles ``(h_min_dcf, h_act_dcf)``.
        """
        h_min = self.plot_min_dcf(color=color)
        h_act = self.plot_act_dcf(color=color)
        return h_min, h_act

    def save(self, img_path: str, dpi: Optional[int] = None, **kwargs: Any) -> None:
        """Saves the current normalized DCF figure.

        Args:
            img_path: Output image path (for example, ``.png`` or ``.pdf``).
            dpi: Optional output DPI.
            **kwargs: Extra keyword arguments passed to ``matplotlib`` ``savefig``.
        """
        save_kwargs = dict(kwargs)
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        self.fh.savefig(img_path, **save_kwargs)

    def plot_operating_point(
        self,
        value: float,
        color: str = "k",
        line_type: str = "--",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots a vertical operating-point line on the DCF plot.

        Args:
            value: X-axis position in logit-prior domain.
            color: Matplotlib color specification.
            line_type: Matplotlib line style specification.
            legend: Optional legend string (no system-name prefix added).

        Returns:
            Any: Matplotlib line handle, or ``None`` if outside x-range.
        """
        value = float(value)
        xmin, xmax = self.plot_axes[0], self.plot_axes[1]
        if value < xmin or value > xmax:
            warnings.warn(
                f"Operating point of {value:.6f} is not between {xmin:.6f} and "
                f"{xmax:.6f}. The line will not be plotted.",
                stacklevel=2,
            )
            return None

        ymin, ymax = self.plot_axes[2], self.plot_axes[3]
        (h,) = self.ax.plot(
            [value, value],
            [ymin, ymax],
            color=color,
            linestyle=line_type,
            label=legend if legend else "_nolegend_",
        )
        if legend:
            self.handles_vec.append(h)
            self.legend_strings.append(legend)
            self.ax.legend(self.handles_vec, self.legend_strings)
        return h

    def plot_dr30_fa(
        self,
        color: Optional[str] = None,
        marker: str = "o",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots the DR30 false-alarm point on the minimum DCF curve.

        Args:
            color: Marker color. If None, uses the current system color.
            marker: Matplotlib marker style.
            legend: Optional legend string. If None, defaults to
                ``"DR30 Pfa {system_name}"``.

        Returns:
            Any: Matplotlib handle for the plotted point, or ``None`` if unavailable.
        """
        if self.minDCF is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        if self.dr30FA is None:
            warnings.warn(
                "DR30 Pfa point is unavailable for current system; nothing will be plotted.",
                stacklevel=2,
            )
            return None

        c = self._resolve_color(color)
        legend = self._resolve_legend(legend, "DR30 Pfa")
        idx = int(self.dr30FA)
        (h,) = self.ax.plot(
            self.plo[idx],
            self.minDCF[idx],
            color=c,
            marker=marker,
            linestyle="None",
            label=legend if legend else "_nolegend_",
        )
        if legend:
            self.handles_vec.append(h)
            self.legend_strings.append(legend)
            self.ax.legend(self.handles_vec, self.legend_strings)
        return h

    def plot_dr30_pmiss(
        self,
        color: Optional[str] = None,
        marker: str = "s",
        legend: Optional[str] = None,
    ) -> Any:
        """Plots the DR30 miss point on the minimum DCF curve.

        Args:
            color: Marker color. If None, uses the current system color.
            marker: Matplotlib marker style.
            legend: Optional legend string. If None, defaults to
                ``"DR30 Pmiss {system_name}"``.

        Returns:
            Any: Matplotlib handle for the plotted point, or ``None`` if unavailable.
        """
        if self.minDCF is None:
            raise ValueError("Call set_system_from_scores or set_system_from_trials first")
        if self.dr30Miss is None:
            warnings.warn(
                "DR30 Pmiss point is unavailable for current system; nothing will be plotted.",
                stacklevel=2,
            )
            return None

        c = self._resolve_color(color)
        legend = self._resolve_legend(legend, "DR30 Pmiss")
        idx = int(self.dr30Miss)
        (h,) = self.ax.plot(
            self.plo[idx],
            self.minDCF[idx],
            color=c,
            marker=marker,
            linestyle="None",
            label=legend if legend else "_nolegend_",
        )
        if legend:
            self.handles_vec.append(h)
            self.legend_strings.append(legend)
            self.ax.legend(self.handles_vec, self.legend_strings)
        return h

    def plot_dr30(self, color: Optional[str] = None) -> Any:
        """Plots both DR30 points (false alarm and miss).

        Args:
            color: Optional shared color for both points.

        Returns:
            Any: Tuple with handles ``(h_dr30_fa, h_dr30_pmiss)``.
        """
        h_fa = self.plot_dr30_fa(color=color)
        h_pmiss = self.plot_dr30_pmiss(color=color)
        return h_fa, h_pmiss
