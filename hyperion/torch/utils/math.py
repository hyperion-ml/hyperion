"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Callable, Union

import torch

LinearSolveFn = Callable[[torch.Tensor], torch.Tensor]
InvertTriMatReturn = Union[LinearSolveFn, list[Union[LinearSolveFn, torch.Tensor]]]


def invert_trimat(
    A: torch.Tensor,
    lower: bool = False,
    right_inv: bool = False,
    return_logdet: bool = False,
    return_inv: bool = False,
) -> InvertTriMatReturn:
    """Inversion of triangular matrices.
       Returns a function that multiplies a matrix of vectors by the inverse of ``A``.

    Args:
      A: Square triangular matrix.
      lower: if True A is lower triangular, else A is upper triangular.
      right_inv: If False, f(v)=A^{-1}v; if True f(v)=v' A^{-1}
      return_logdet: If True, it also returns the log determinant of A.
      return_inv: If True, it also returns A^{-1}

    Returns:
      If ``return_logdet`` and ``return_inv`` are both False, it returns the
      solve function ``f``.
      Otherwise, it returns a list starting with ``f`` and optionally followed by
      ``logdet(A)`` and ``A^{-1}`` depending on ``return_logdet`` and
      ``return_inv``.
    """

    if right_inv:
        fh = lambda x: torch.linalg.solve_triangular(A.t(), x.t(), upper=lower).t()
    else:
        fh = lambda x: torch.linalg.solve_triangular(A, x, upper=not (lower))

    if return_logdet or return_inv:
        r = [fh]
    else:
        r = fh

    if return_logdet:
        logdet = torch.sum(torch.log(torch.diag(A)))
        r.append(logdet)

    if return_inv:
        invA = fh(torch.eye(A.shape[0], device=A.device, dtype=A.dtype))
        r.append(invA)

    return r
