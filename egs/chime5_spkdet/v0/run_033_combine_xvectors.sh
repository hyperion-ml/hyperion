#!/bin/bash
# Copyright 2019   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_name=3b
diar_name=diar3b_t-0.9
track_name=track3b_t-0.9
stage=1

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors/$net_name

if [ $stage -le 1 ]; then

    utils/combine_data.sh data/chime5_spkdet data/chime5_spkdet_enroll data/chime5_spkdet_test
    mkdir -p $xvector_dir/chime5_spkdet
    cat $xvector_dir/chime5_spkdet_{enroll,test}/xvector.scp > $xvector_dir/chime5_spkdet/xvector.scp

    utils/combine_data.sh data/chime5_spkdet_gtvad data/chime5_spkdet_enroll data/chime5_spkdet_test_gtvad
    mkdir -p $xvector_dir/chime5_spkdet_gtvad
    cat $xvector_dir/chime5_spkdet_{enroll,test_gtvad}/xvector.scp > $xvector_dir/chime5_spkdet_gtvad/xvector.scp

fi

if [ $stage -le 2 ]; then

    utils/combine_data.sh data/chime5_spkdet_${diar_name} data/chime5_spkdet_enroll data/chime5_spkdet_test_${diar_name}
    mkdir -p $xvector_dir/chime5_spkdet_${diar_name}
    cat $xvector_dir/chime5_spkdet_{enroll,test_${diar_name}}/xvector.scp > $xvector_dir/chime5_spkdet_${diar_name}/xvector.scp

fi

if [ $stage -le 3 ]; then

    utils/combine_data.sh data/chime5_spkdet_${track_name} data/chime5_spkdet_enroll data/chime5_spkdet_test_${track_name}
    mkdir -p $xvector_dir/chime5_spkdet_${track_name}
    cat $xvector_dir/chime5_spkdet_{enroll,test_${track_name}}/xvector.scp > $xvector_dir/chime5_spkdet_${track_name}/xvector.scp

fi


exit
