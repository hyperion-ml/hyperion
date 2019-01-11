"""
Utility functions to evaluate performance
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np



def pavx(y):

#PAV: Pool Adjacent Violators algorithm. Non-paramtetric optimization subject to monotonicity.
#
# ghat = pav(y)
# fits a vector ghat with nondecreasing components to the 
# data vector y such that sum((y - ghat).^2) is minimal. 
# (Pool-adjacent-violators algorithm).
#
# optional outputs:
#   width: width of pav bins, from left to right 
#          (the number of bins is data dependent)
#   height: corresponding heights of bins (in increasing order)
# 
# Author: This code is a simplified version of the 'IsoMeans.m' code made available 
# by Lutz Duembgen at:
# http://www.imsv.unibe.ch/~duembgen/software

    assert(isinstance(y, np.ndarray))

    n = len(y)
    assert(n>0)
    index = np.zeros(y.shape, dtype=int)
    l = np.zeros(y.shape, dtype=int)
    # An interval of indices is represented by its left endpoint 
    # ("index") and its length "len" 
    ghat = np.zeros_like(y)

    ci = 0
    index[ci] = 0
    l[ci] = 1
    ghat[ci] = y[0]
    # ci is the number of the interval considered currently.
    # ghat[ci] is the mean of y-values within this interval.
    for j in xrange(1, n):
        # a new index intervall, {j}, is created:
        ci = ci+1
        index[ci] = j
        l[ci] = 1
        ghat[ci] = y[j]
        #while ci >= 1 and ghat[np.maximum(ci-1,0)] >= ghat[ci]:
        while ci >= 1 and ghat[ci-1] >= ghat[ci]:
            # "pool adjacent violators":
            nw = l[ci-1] + l[ci]
            ghat[ci-1] = ghat[ci-1] + (l[ci] / nw) * (
                ghat[ci] - ghat[ci-1])
            l[ci-1] = nw
            ci = ci-1

    
    height = np.copy(ghat[:ci+1])
    width = l[:ci+1]

    # Now define ghat for all indices:
    while n >= 1:
        for j in xrange(index[ci], n):
            ghat[j] = ghat[ci]

        n = index[ci]
        ci = ci-1

    return ghat, width, height


