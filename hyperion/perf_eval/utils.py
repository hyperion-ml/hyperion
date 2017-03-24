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

    n = len(y);
    assert(n>0)
    index = np.zeros(y.shape, dtype=int);
    l = np.zeros(y.shape, dtype=int);
    # An interval of indices is represented by its left endpoint 
    # ("index") and its length "len" 
    ghat = np.zeros_like(y);

    ci = 0;
    index[ci] = 0;
    l[ci] = 1;
    ghat[ci] = y[0];
    # ci is the number of the interval considered currently.
    # ghat[ci] is the mean of y-values within this interval.
    for j in xrange(1, n):
        # a new index intervall, {j}, is created:
        ci = ci+1;
        index[ci] = j;
        l[ci] = 1;
        ghat[ci] = y[j];
        while ci >= 2 && ghat[np.maximum(ci-1,1)] >= ghat[ci]:
            # "pool adjacent violators":
            nw = l[ci-1] + l[ci];
            ghat[ci-1] = ghat[ci-1] + (l[ci] / nw) * (
                ghat[ci] - ghat[ci-1]);
            l[ci-1] = nw;
            ci = ci-1;



    height = ghat[:ci];
    width = l[:ci];

    # Now define ghat for all indices:
    while n >= 0:
        for j in xrange(index[ci], n):
            ghat[j] = ghat[ci];

        n = index[ci]-1;
        ci = ci-1;



def compute_rocch(tar_scores, non_scores):
# ROCCH: ROC Convex Hull.
# Usage: [pmiss,pfa] = rocch(tar_scores,nontar_scores)
# (This function has the same interface as compute_roc.)
#
# Note: pmiss and pfa contain the coordinates of the vertices of the
#       ROC Convex Hull.
#
# For a demonstration that plots ROCCH against ROC for a few cases, just
# type 'rocch' at the MATLAB command line.
#
# Inputs:
#   tar_scores: scores for target trials
#   nontar_scores: scores for non-target trials


    assert(isinstance(tar_scores, np.ndarray))
    assert(isinstance(non_scores, np.ndarray))
    
    Nt = len(tar_scores);
    Nn = len(non_scores);
    N = Nt+Nn;
    scores = np.hstack((tar_scores.ravel(), non_scores.ravel()))
    #ideal, but non-monotonic posterior
    Pideal = np.hstack((np.ones((Nt,)), np.zeros((Nn,))))
                    
    #It is important here that scores that are the same (i.e. already in order) should NOT be swapped.
    #MATLAB's sort algorithm has this property.
    perturb = np.argsort(scores);
                    
    Pideal = Pideal[perturb];
    Popt, width = pavx(Pideal); 

    nbins = len(width);
    pmiss = np.zeros((nbins+1,));
    pfa = np.zeros((nbins+1,));

    #threshold leftmost: accept eveything, miss nothing
    left = 0; #0 scores to left of threshold
    fa = Nn;
    miss = 0;

    for i in xrange(nbins):
        pmiss[i] = miss/Nt;
        pfa[i] = fa/Nn;
        left = left + width[i];
        miss = np.sum(Pideal[:left]);
        fa = N - left - np.sum(Pideal[left:]);

    pmiss[nbins] = miss/Nt;
    pfa[nbins] = fa/Nn;





