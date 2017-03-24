from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np


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



def test_roc():
    figure();

subplot(2,3,1);
tar = [1]; non = [0];
[pmiss,pfa] = rocch(tar,non);
[pm,pf] = compute_roc(tar,non);
plot(pfa,pmiss,'r-^',pf,pm,'g--v');
axis('square');grid;legend('ROCCH','ROC');
title('2 scores: non < tar');

subplot(2,3,2);
tar = [0]; non = [1];
[pmiss,pfa] = rocch(tar,non);
[pm,pf] = compute_roc(tar,non);
plot(pfa,pmiss,'r-^',pf,pm,'g-v');
axis('square');grid;
title('2 scores: tar < non');

subplot(2,3,3);
tar = [0]; non = [-1,1];
[pmiss,pfa] = rocch(tar,non);
[pm,pf] = compute_roc(tar,non);
plot(pfa,pmiss,'r-^',pf,pm,'g--v');
axis('square');grid;
title('3 scores: non < tar < non');

subplot(2,3,4);
tar = [-1,1]; non = [0];
[pmiss,pfa] = rocch(tar,non);
[pm,pf] = compute_roc(tar,non);
plot(pfa,pmiss,'r-^',pf,pm,'g--v');
axis('square');grid;
title('3 scores: tar < non < tar');
xlabel('P_{fa}');
ylabel('P_{miss}');

subplot(2,3,5);
tar = randn(1,100)+1; non = randn(1,100);
[pmiss,pfa] = rocch(tar,non);
[pm,pf] = compute_roc(tar,non);
plot(pfa,pmiss,'r-^',pf,pm,'g');
axis('square');grid;
title('45^{\circ} DET');

subplot(2,3,6);
tar = randn(1,100)*2+1; non = randn(1,100);
[pmiss,pfa] = rocch(tar,non);
[pm,pf] = compute_roc(tar,non);
plot(pfa,pmiss,'r-^',pf,pm,'g');
axis('square');grid;
title('flatter DET');

end
