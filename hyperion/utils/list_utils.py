"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Utilities for lists.
"""

import numpy as np
from operator import itemgetter
from itertools import groupby


def list2ndarray(a, dtype=None):
    """Converts python string list to string numpy array."""
    if isinstance(a, list):
        return np.asarray(a, dtype=dtype)
    assert isinstance(a, np.ndarray)
    return a


def ismember(a, b):
    """Replicates MATLAB ismember function.

    Returns:
      For or arrays A and B returns an array of the same
      size as A containing true where the elements of A are in B and false
      otherwise.

      Also returns an array LOC containing the
      lowest absolute index in B for each element in A which is a member of
      B and 0 if there is no such index.
    """
    bad_idx = np.iinfo(np.int32).min
    d = {}
    for i, elt in enumerate(b):
        if elt not in d:
            d[elt] = i
    loc = np.array([d.get(x, bad_idx) for x in a], dtype="int32")
    f = loc != bad_idx
    return f, loc


def sort(a, reverse=False, return_index=False):
    """Sorts a list or numpy array

    Args:
     a: string list or numpy array to sort.
     reverse: it True it sorts from high to low, otherwise from low to high.
     return_index: It true it returns numpy array with the indices of the
                   elements of a in the sorted array.
    Returns:
      Sorted numpy array.
      Index numpy array if return_index is True
    """
    if return_index:
        idx = np.argsort(a)
        if reverse:
            idx = idx[::-1]
        if not (isinstance(a, np.ndarray)):
            a = np.asarray(a)
        s_a = a[idx]
        return s_a, idx
    else:
        if reverse:
            return np.sort(a)[::-1]
        return np.sort(a)


def intersect(a, b, assume_unique=False, return_index=False):
    """Computes the interseccion of a and b lists or numpy arrays.

    Args:
      a: First list to intersect.
      b: Second list to intersect.
      assume_unique: If True, the input arrays are both assumed to be unique,
                     which can speed up the calculation. Default is False.
      return_index: if True, it returns two numpy arrays with:
                     - the indeces of the elements of a that are in a and b.
                     - the indeces of the elements of b that are in a and b.

    Returns:
      c: numpy array with the interseccion of a and b.
      ia: indeces of a in c if return_index is True
      ib: indeces of b in c if return_index is True
    """
    c = np.intersect1d(a, b, assume_unique)
    if return_index:
        _, ia = ismember(c, a)
        _, ib = ismember(c, b)
        return c, ia, ib
    else:
        return c


def split_list(a, idx, num_parts):
    """Split a list into several parts and returns one of the parts.

    Args:
       a: list to split.
       idx: index of the part that we want to get from 1 to num_parts
       num_parts: number of parts to split the list.

    Returns:
       A sublist of a.
    """
    if not (isinstance(a, np.ndarray)):
        a = np.asarray(a)
    n = float(len(a))
    idx_1 = int(np.floor((idx - 1) * n / num_parts))
    idx_2 = int(np.floor(idx * n / num_parts))
    loc = np.arange(idx_1, idx_2, dtype="int64")
    return a[loc], loc


def split_list_group_by_key(a, idx, num_parts, key=None):
    """Split a list into several parts and returns one of the parts.
       It groups the elements of a with the same key into the same part.
    Args:
       a: list to split.
       idx: index of the part that we want to get from 1 to num_parts
       num_parts: number of parts to split the list.
       key: List of properties of a, it groups the elements of a with the
            same key into the same part.

    Returns:
       A sublist of a.
    """

    if not (isinstance(a, np.ndarray)):
        a = np.asarray(a)
    if key is None:
        key = a
    _, ids = np.unique(key, return_inverse=True)
    n = float(ids.max() + 1)
    idx_1 = int(np.floor((idx - 1) * n / num_parts))
    idx_2 = int(np.floor(idx * n / num_parts))
    loc = np.empty(len(a), dtype="int64")
    k = 0
    for i in range(idx_1, idx_2):
        loc_i = (ids == i).nonzero()[0]
        loc[k : k + len(loc_i)] = loc_i
        k += len(loc_i)
    loc = loc[:k]
    return a[loc], loc
