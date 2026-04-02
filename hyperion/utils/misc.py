"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Miscellaneous functions
"""

import logging
import shutil
import typing
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, TypeVar, Union

import matplotlib as mpl
import numpy as np
import scipy.sparse as sparse

PathLike = Union[str, Path]
ArrayLike = Union[float, np.ndarray]
_latex_checked: Optional[bool] = None


def generate_data(g: typing.Any) -> typing.Any:
    while 1:
        yield g.get_next_batch()


def str2bool(s: str) -> typing.Any:
    """Convert string to bool for argparse"""
    if isinstance(s, bool):
        return s

    values = {
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    if s.lower() not in values:
        raise ValueError("Need bool; got %r" % s)
    return values[s.lower()]


def apply_gain_logx(x: ArrayLike, AdB: ArrayLike) -> ArrayLike:
    """
    Apply a gain in decibels (dB) to a log-amplitude representation `log(x)`.

    This function assumes the input `x` is log-amplitude (i.e., log(x)), and applies the gain linearly in that domain.

    Parameters:
        x (float or np.ndarray): Log-amplitude input.
        AdB (float): Gain to apply in decibels.

    Returns:
        float or np.ndarray: Gain-adjusted log-amplitude.
    """
    return x + AdB / (20.0 * np.log10(np.exp(1)))


def apply_gain_logx2(x: ArrayLike, AdB: ArrayLike) -> ArrayLike:
    """
    Apply a gain in decibels (dB) to a log-power representation `log(x^2)`.

    This function assumes the input `x` is log-power (i.e., log(x^2)), and applies the gain linearly in that domain.

    Parameters:
        x (float or np.ndarray): Log-power input.
        AdB (float): Gain to apply in decibels.

    Returns:
        float or np.ndarray: Gain-adjusted log-power.
    """
    return x + AdB / (10.0 * np.log10(np.exp(1)))


def apply_gain_x(x: ArrayLike, AdB: ArrayLike) -> ArrayLike:
    """
    Apply a gain in decibels (dB) to a linear amplitude signal `x`.

    Parameters:
        x (float or np.ndarray): Linear amplitude input.
        AdB (float): Gain to apply in decibels.

    Returns:
        float or np.ndarray: Gain-adjusted amplitude.
    """
    return x * 10 ** (AdB / 20)


def apply_gain_x2(x: ArrayLike, AdB: ArrayLike) -> ArrayLike:
    """
    Apply a gain in decibels (dB) to a power signal `x^2`.

    Parameters:
        x (float or np.ndarray): Power input (e.g., amplitude squared).
        AdB (float): Gain to apply in decibels.

    Returns:
        float or np.ndarray: Gain-adjusted power.
    """
    return x * 10 ** (AdB / 10)


def apply_gain(x: ArrayLike, feat_type: str, AdB: ArrayLike) -> ArrayLike:

    f_dict = {
        "fft": apply_gain_x,
        "logfft": apply_gain_logx,
        "fb": apply_gain_x,
        "fb2": apply_gain_x2,
        "logfb": apply_gain_logx,
        "logfb2": apply_gain_logx2,
    }
    if feat_type not in f_dict:
        valid = ", ".join(sorted(f_dict.keys()))
        raise ValueError(
            f"Unknown feat_type={feat_type!r}. Supported values: {valid}"
        )
    f = f_dict[feat_type]
    return f(x, AdB)


def energy_vad(P: np.ndarray) -> np.ndarray:
    """
    Perform simple energy-based Voice Activity Detection (VAD).

    Marks frames as active where power is within 35 dB of the maximum power.

    Parameters:
        P (np.ndarray): Power or energy over time.

    Returns:
        np.ndarray (bool): Boolean mask of voiced frames.
    """
    thr = np.max(P) - 35
    return P > thr


def compute_snr(
    x: np.ndarray, n: np.ndarray, axis: int = -1
) -> Union[float, np.ndarray]:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels between `x` and `n`.

    Parameters:
        x (np.ndarray): Signal waveform or feature.
        n (np.ndarray): Noise waveform or feature.
        axis (int): Axis along which to compute the mean power.

    Returns:
        float or np.ndarray: SNR in decibels.
    """
    P_x = 10 * np.log10(np.mean(x**2, axis=axis))
    P_n = 10 * np.log10(np.mean(n**2, axis=axis))
    return P_x - P_n


def filter_args(valid_args: Iterable[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters arguments from a dictionary.

    Args:
        valid_args: Iterable of valid argument names.
        kwargs: Dictionary containing program config arguments.

    Returns:
        Dictionary with only keys from valid_args if they exist in kwargs.
    """
    return dict((k, kwargs[k]) for k in valid_args if k in kwargs)


def filter_func_args(
    func: Callable[..., Any], kwargs: Dict[str, Any], skip: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Filters arguments expected by a function.

    Args:
        func: Target function object.
        kwargs: Dictionary containing arguments.
        skip: Set of argument names to exclude (e.g., "self").

    Returns:
        Dictionary with arguments expected by the target function.
    """
    sig = signature(func)
    skip_args = set() if skip is None else set(skip)
    skip_args.add("self")

    valid_args = [k for k in sig.parameters.keys() if k not in skip_args]
    my_kwargs = filter_args(valid_args, kwargs)

    if "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
        my_kwargs.update(filter_args(valid_args, kwargs["kwargs"]))

    args = sig.bind_partial(**my_kwargs).arguments
    return args


from tqdm import tqdm


def tqdm_urlretrieve_hook(
    t: tqdm,
) -> Callable[[int, int, Optional[int]], Optional[int]]:
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> from urllib.request import urlretrieve
    >>> with tqdm(...) as t:
    ...     reporthook = tqdm_urlretrieve_hook(t)
    ...     urlretrieve(..., reporthook=reporthook)
    Source: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """
    last_b = [0]

    def update_to(
        b: int = 1, bsize: int = 1, tsize: Optional[int] = None
    ) -> Optional[int]:
        """
        Update tqdm progress bar.

        Args:
            b: Number of blocks transferred so far [default: 1].
            bsize: Size of each block in tqdm units [default: 1].
            tsize: Total size in tqdm units. If None or -1, remains unchanged.

        Returns:
            Number of bytes updated (or None).
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to


def urlretrieve_progress(
    url: str,
    filename: Optional[str] = None,
    data: Optional[Any] = None,
    desc: Optional[str] = None,
) -> Any:
    """
    Works like urllib.request.urlretrieve, but displays a tqdm progress bar during download.
    Taken from lhotse: https://github.com/lhotse-speech/lhotse/blob/master/lhotse/utils.py

    Args:
        url: URL to download.
        filename: Optional path to save the file.
        data: Optional POST data to send.
        desc: Optional description for the tqdm progress bar.

    Returns:
        The result of urllib.request.urlretrieve.
    """
    from urllib.request import urlretrieve

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)


def build_class_labels_from_boolean_matrix_dense(B: np.ndarray) -> typing.Any:
    """
    Given a boolean matrix B of shape (N, M), where B[i, j] is True if row i and column j
    are of the same class, this function returns class labels for rows and columns.

    Parameters:
        B (np.ndarray): A 2D boolean array of shape (N, M)

    Returns:
        row_labels (np.ndarray): Class IDs for each row (shape: N,)
        col_labels (np.ndarray): Class IDs for each column (shape: M,)
    """
    B = B.astype(bool)
    N, M = B.shape

    # Create an (N+M) x (N+M) adjacency matrix
    adj = np.zeros((N + M, N + M), dtype=bool)

    # Set connections between row i and column j (offset by N)
    row_idx, col_idx = np.where(B)
    for i, j in zip(row_idx, col_idx):
        adj[i, N + j] = True
        adj[N + j, i] = True  # Make it undirected

    # Find connected components in the undirected graph
    n_components, labels = sparse.csgraph.connected_components(adj, directed=False)

    row_labels = labels[:N]
    col_labels = labels[N:]

    return row_labels, col_labels


def build_class_labels_from_boolean_matrix_sparse(B: sparse.csr_matrix) -> typing.Any:
    """
    Given a boolean sparse matrix B of shape (N, M), where B[i, j] is True if row i and column j
    are of the same class, this function returns class labels for rows and columns.

    Parameters:
        B (sparse.csr_matrix): A 2D boolean sparse matrix of shape (N, M)

    Returns:
        row_labels (np.ndarray): Class IDs for each row (shape: N,)
        col_labels (np.ndarray): Class IDs for each column (shape: M,)
    """
    N, M = B.shape
    # Build bipartite adjacency matrix: rows [0..N-1], cols [N..N+M-1]
    # Upper right: B, Lower left: B.T
    top = sparse.hstack([sparse.csr_matrix((N, N)), B])
    bottom = sparse.hstack([B.transpose(), sparse.csr_matrix((M, M))])
    adj = sparse.vstack([top, bottom])

    # Make the graph undirected
    adj = adj + adj.transpose()

    n_components, labels = sparse.csgraph.connected_components(adj, directed=False)

    row_labels = labels[:N]
    col_labels = labels[N:]
    return row_labels, col_labels


def check_and_disable_latex() -> None:
    global _latex_checked
    if not mpl.rcParams.get("text.usetex", False):
        return

    if _latex_checked is False:
        mpl.rcParams["text.usetex"] = False
        return

    if shutil.which("latex") is None:
        logging.warning("LaTeX not found. Disabling `usetex`.")
        mpl.rcParams["text.usetex"] = False
        _latex_checked = False
        return

    if _latex_checked:
        return

    try:
        from matplotlib import texmanager

        texmanager.TexManager.get_text_width_height_descent("lp", 10)
        _latex_checked = True
    except Exception as e:
        # Latex can be present in PATH but still unusable (missing TeX packages).
        err = str(e).splitlines()[0]
        logging.warning("LaTeX check failed (%s). Disabling `usetex`.", err)
        mpl.rcParams["text.usetex"] = False
        _latex_checked = False


def check_spd_mat(name: typing.Any, A: typing.Any) -> None:
    # symmetry
    sym_err = np.linalg.norm(A - A.T, ord="fro") / (
        np.linalg.norm(A, ord="fro") + 1e-12
    )
    # eigenvalues
    w = np.linalg.eigvalsh((A + A.T) / 2)
    w_min = w[0]
    w_max = w[-1]
    # conditioning
    cond = w_max / max(w_min, 1e-12)
    logging.info(
        f"{name}: sym_err={sym_err:.3e}  w_min={w_min:.3e}  w_max={w_max:.3e}  cond={cond:.3e}"
    )
