#!/usr/bin/env python

'''
Run VAE
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_hist


def plot_val_loss(in_files, out_file, title):

    plt.figure(figsize=(12, 6))
    plt.hold(True)

    cc = ['b', 'r', 'g', 'm', 'c', 'k']
    hh = []
    labels = []
    for i in xrange(len(in_files)):
        label, loss, val_loss = load_hist(in_files[i])
        h_i, = plt.plot(-val_loss, c=cc[i], label='VAE')
        hh.append(h_i)
        labels.append(label)

    plt.legend(hh, labels)
    plt.ylabel('Val loss')
    plt.xlabel('epoch')
    plt.title(title)
    plt.savefig(out_file)
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Merge val losses in one plot')

    parser.add_argument('--title', dest='title', required=True)
    parser.add_argument('--in-files', dest='in_files', nargs='+', required=True)
    parser.add_argument('--out-file', dest='out_file', required=True)
    args=parser.parse_args()

    plot_val_loss(args.in_files, args.out_file, args.title)
    
