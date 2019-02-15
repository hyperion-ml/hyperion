#!/bin/bash

# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e


stage=1

. parse_options.sh || exit 1;

#If you have already prepared the data for v1.8k recipe you can just copy it and save the data preparation.

# Get data from v1.8k
rsync -rptlv ../v1.8k/data .

# Clean some unnecessary files
find data/ -name "split*" | xargs rm -rf
find data/ -name ".backup" | xargs rm -rf

