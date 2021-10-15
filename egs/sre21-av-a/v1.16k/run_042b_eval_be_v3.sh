#!/bin/bash

# Tuned for res2net s=4 trained on voxceleb

echo ./run_042b_eval_be_v3.sh --pca-var-r 0.85 --r-mu 100 --w-mu1 0.5 --w-B1 0.25 --w-W1 0.25 --w-mu2 0.75 --w-B2 0.25 --w-W2 0.25 "$@"
./run_042_eval_be_v3.sh --pca-var-r 0.85 --r-mu 100 --w-mu1 0.5 --w-B1 0.25 --w-W1 0.25 --w-mu2 0.75 --w-B2 0.25 --w-W2 0.25 "$@"
