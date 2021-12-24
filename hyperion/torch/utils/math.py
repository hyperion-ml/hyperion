"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch


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
        fh = lambda x: torch.triangular_solve(x.t(), A.t(), upper=lower)[0].t()
    else:
        fh = lambda x: torch.triangular_solve(x, A, upper=not (lower))[0]

    if return_logdet or return_inv:
        r = [fh]
    else:
        r = fh

    if return_logdet:
        logdet = torch.sum(torch.log(torch.diag(A)))
        r.append(logdet)

    if return_inv:
        invA = fh(torch.eye(A.shape[0]))
        r.append(invA)

    return r
