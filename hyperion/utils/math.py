"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Some math functions.
"""

import numpy as np
import scipy.linalg as la

from ..hyp_defs import float_cpu


def logdet_pdmat(A):
    """Log determinant of positive definite matrix."""
    assert A.shape[0] == A.shape[1]
    R = la.cholesky(A)
    return 2 * np.sum(np.log(np.diag(R)))


def invert_pdmat(A, right_inv=False, return_logdet=False, return_inv=False):
    """Inversion of positive definite matrices.
       Returns lambda function f that multiplies the inverse of A times a vector.

    Args:
      A: Positive definite matrix
      right_inv: If False, f(v)=A^{-1}v; if True f(v)=v' A^{-1}
      return_logdet: If True, it also returns the log determinant of A.
      return_inv: If True, it also returns A^{-1}

    Returns:
      Lambda function that multiplies A^{-1} times vector.
      Cholesky transform of A upper triangular
      Log determinant of A
      A^{-1}
    """
    assert A.shape[0] == A.shape[1]
    R = la.cholesky(A, lower=False)

    if right_inv:
        fh = lambda x: la.cho_solve((R, False), x.T).T
    else:
        fh = lambda x: la.cho_solve((R, False), x)
        # fh=lambda x: la.solve_triangular(R, la.solve_triangular(R.T, x, lower=True), lower=False)

    r = [fh, R]

    logdet = None
    invA = None

    if return_logdet:
        logdet = 2 * np.sum(np.log(np.diag(R)))
        r.append(logdet)

    if return_inv:
        invA = fh(np.eye(A.shape[0]))
        r.append(invA)

    return r


def invert_trimat(
    A, lower=False, right_inv=False, return_logdet=False, return_inv=False
):
    """Inversion of triangular matrices.
       Returns lambda function f that multiplies the inverse of A times a vector.

    Args:
      A: Triangular matrix.
      lower: if True A is lower triangular, else A is upper triangular.
      right_inv: If False, f(v)=A^{-1}v; if True f(v)=v' A^{-1}
      return_logdet: If True, it also returns the log determinant of A.
      return_inv: If True, it also returns A^{-1}

    Returns:
      Lambda function that multiplies A^{-1} times vector.
      Log determinant of A
      A^{-1}
    """

    if right_inv:
        fh = lambda x: la.solve_triangular(A.T, x.T, lower=not (lower)).T
    else:
        fh = lambda x: la.solve_triangular(A, x, lower=lower)

    if return_logdet or return_inv:
        r = [fh]
    else:
        r = fh

    if return_logdet:
        logdet = np.sum(np.log(np.diag(A)))
        r.append(logdet)

    if return_inv:
        invA = fh(np.eye(A.shape[0]))
        r.append(invA)

    return r


def softmax(r, axis=-1):
    """
    Returns:
      y = \exp(r)/\sum(\exp(r))
    """
    max_r = np.max(r, axis=axis, keepdims=True)
    r = np.exp(r - max_r)
    r /= np.sum(r, axis=axis, keepdims=True)
    return r


def logsumexp(r, axis=-1):
    """
    Returns:
      y = \log \sum(\exp(r))
    """
    max_r = np.max(r, axis=axis, keepdims=True)
    r = np.exp(r - max_r)
    return np.log(np.sum(r, axis=axis) + 1e-20) + np.squeeze(max_r, axis=axis)


def logsigmoid(x):
    """
    Returns:
      y = \log(sigmoid(x))
    """
    e = np.exp(-x)
    f = x < -100
    log_p = -np.log(1 + np.exp(-x))
    log_p[f] = x[f]
    return log_p


def neglogsigmoid(x):
    """
    Returns:
      y = -\log(sigmoid(x))
    """
    e = np.exp(-x)
    f = x < -100
    log_p = np.log(1 + np.exp(-x))
    log_p[f] = -x[f]
    return log_p


def sigmoid(x):
    """
    Returns:
      y = sigmoid(x)
    """
    e = np.exp(-x)
    f = x < -100
    p = 1 / (1 + np.exp(-x))
    p[f] = 0
    return p


def fisher_ratio(mu1, Sigma1, mu2, Sigma2):
    """Computes the Fisher ratio between two classes
    from the class means and covariances.
    """
    S = Sigma1 + Sigma2
    L = invert_pdmat(S)[0]
    delta = mu1 - mu2
    return np.inner(delta, L(delta))


def fisher_ratio_with_precs(mu1, Lambda1, mu2, Lambda2):
    """Computes the Fisher ratio between two classes
    from the class means precisions.
    """

    Sigma1 = invert_pdmat(Lambda1, return_inv=True)[-1]
    Sigma2 = invert_pdmat(Lambda2, return_inv=True)[-1]
    return fisher_ratio(mu1, Sigma1, mu2, Sigma2)


def symmat2vec(A, lower=False, diag_factor=None):
    """Puts a symmetric matrix into a vector.

    Args:
      A: Symmetric matrix.
      lower: If True, it uses the lower triangular part of the matrix.
             If False, it uses the upper triangular part of the matrix.
      diag_factor: It multiplies the diagonal of A by diag_factor.

    Returns:
      Vector with the upper or lower triangular part of A.
    """
    if diag_factor is not None:
        A = np.copy(A)
        A[np.diag_indices(A.shape[0])] *= diag_factor
    if lower:
        return A[np.tril_indices(A.shape[0])]
    return A[np.triu_indices(A.shape[0])]


def vec2symmat(v, lower=False, diag_factor=None):
    """Puts a vector back into a symmetric matrix.

    Args:
      v: Vector with the upper or lower triangular part of A.
      lower: If True, v contains the lower triangular part of the matrix.
             If False, v contains the upper triangular part of the matrix.
      diag_factor: It multiplies the diagonal of A by diag_factor.

    Returns:
      Symmetric matrix.
    """

    dim = int((-1 + np.sqrt(1 + 8 * v.shape[0])) / 2)
    idx_u = np.triu_indices(dim)
    idx_l = np.tril_indices(dim)
    A = np.zeros((dim, dim), dtype=float_cpu())
    if lower:
        A[idx_l] = v
        A[idx_u] = A.T[idx_u]
    else:
        A[idx_u] = v
        A[idx_l] = A.T[idx_l]
    if diag_factor is not None:
        A[np.diag_indices(A.shape[0])] *= diag_factor
    return A


def trimat2vec(A, lower=False):
    """Puts a triangular matrix into a vector.

    Args:
      A: Triangular matrix.
      lower: If True, it uses the lower triangular part of the matrix.
             If False, it uses the upper triangular part of the matrix.

    Returns:
      Vector with the upper or lower triangular part of A.
    """

    return symmat2vec(A, lower)


def vec2trimat(v, lower=False):
    """Puts a vector back into a triangular matrix.

    Args:
      v: Vector with the upper or lower triangular part of A.
      lower: If True, v contains the lower triangular part of the matrix.
             If False, v contains the upper triangular part of the matrix.

    Returns:
      Triangular matrix.
    """
    dim = int((-1 + np.sqrt(1 + 8 * v.shape[0])) / 2)
    A = np.zeros((dim, dim), dtype=float_cpu())
    if lower:
        A[np.tril_indices(dim)] = v
        return A
    A[np.triu_indices(dim)] = v
    return A


def fullcov_varfloor(S, F, F_is_chol=False, lower=False):
    """Variance flooring for full covariance matrices.

    Args:
      S: Covariance.
      F: Minimum cov or Cholesqy decomposisition of it
      F_is_chol: If True F is Cholesqy decomposition
      lower: True if cholF is lower triangular, False otherwise

    Returns:
      Floored covariance
    """
    if isinstance(F, np.ndarray):
        if not F_is_chol:
            cholF = la.cholesky(F, lower=False, overwrite_a=False)
        else:
            cholF = F
            if lower:
                cholF = cholF.T
        icholF = invert_trimat(cholF, return_inv=True)[-1]
        T = np.dot(np.dot(icholF.T, S), icholF)
    else:
        T = S / F

    u, d, _ = la.svd(T, full_matrices=False, overwrite_a=True)
    d[d < 1.0] = 1
    T = np.dot(u * d, u.T)

    if isinstance(F, np.ndarray):
        S = np.dot(cholF.T, np.dot(T, cholF))
    else:
        S = F * T
    return S


def fullcov_varfloor_from_cholS(cholS, cholF, lower=False):
    """Variance flooring for full covariance matrices
       using Cholesky decomposition as input/output

    Args:
      cholS: Cholesqy decomposisition of the covariance.
      cholF: Cholesqy decomposisition of the minimum covariance.
      lower: True if matrices are lower triangular, False otherwise

    Returns:
      Cholesky decomposition of the floored covariance
    """

    if isinstance(cholF, np.ndarray):
        if lower:
            cholS = cholS.T
            cholF = cholF.T
        T = np.dot(cholS, invert_trimat(cholF, return_inv=True)[-1])
    else:
        if lower:
            cholS = cholS.T
        T = cholS / cholF
    T = np.dot(T.T, T)
    u, d, _ = la.svd(T, full_matrices=False, overwrite_a=True)
    d[d < 1.0] = 1
    T = np.dot(u * d, u.T)
    if isinstance(cholF, np.ndarray):
        S = np.dot(cholF.T, np.dot(T, cholF))
    else:
        S = (cholF ** 2) * T
    return la.cholesky(S, lower)


def int2onehot(class_ids, num_classes=None):
    """Integer to 1-hot vector.

    Args:
      class_ids: Numpy array of integers.
      num_classes: Maximum number of classes.

    Returns:
      1-hot Numpy array.
    """

    if num_classes is None:
        num_classes = np.max(class_ids) + 1

    p = np.zeros((len(class_ids), num_classes), dtype=float_cpu())
    p[np.arange(len(class_ids)), class_ids] = 1
    return p


def cosine_scoring(x1, x2):

    l2_1 = np.sqrt(np.sum(x1 ** 2, axis=-1, keepdims=True))
    l2_2 = np.sqrt(np.sum(x2 ** 2, axis=-1, keepdims=True))
    x1 = x1 / l2_1
    x2 = x2 / l2_2

    return np.dot(x1, x2.T)
