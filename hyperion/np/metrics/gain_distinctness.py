"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np


def compute_diag_dominance(M: np.ndarray) -> float:
    """
    Computes the diagonal dominance of a matrix M.

    Diagonal dominance is defined as the sum of the the absolute difference between the averages of the diagonal and the off-diagonal elements

    Args:
        M (np.ndarray): The input matrix.

    Returns:
        float: The diagonal dominance of the matrix.
    """
    M_diag = np.diag(M)
    M_off_diag = M[~np.eye(M.shape[0], dtype=bool)]
    M_diag = M_diag[np.isfinite(M_diag)]
    M_off_diag = M_off_diag[np.isfinite(M_off_diag)]
    diag_dominance = np.abs(M_diag.mean() - M_off_diag.mean())
    return diag_dominance


def compute_gain_distinctness(M_oo: np.ndarray, M_aa: np.ndarray) -> float:
    """
    Computes the gain distinctness given similarity matrix for anonymized and original trials.

    Args:
        M_oo (np.ndarray): The original vs original similarity matrix.
        M_aa (np.ndarray): The anonymized vs anonymized similarity matrix.

    Returns:
        float: The gain distinctness of the matrix.
    """
    diag_dominance_oo = compute_diag_dominance(M_oo)
    diag_dominance_aa = compute_diag_dominance(M_aa)
    print("diagdominance", diag_dominance_aa, diag_dominance_oo, flush=True)
    return 10 * np.log10(diag_dominance_aa / diag_dominance_oo)
