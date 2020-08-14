#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh

be_names_caldev=(lda150_splday125_adapt_v1_a1_mu1B0.75W0.75_a2_M975_mu1B0.6W0.6_sre_tel lda150_splday125_adapt_v2eval_a1_mu1B0.35W1_a2_M500_mu1B0.75W0.8_sre_tel)
be_tags_caldev=('splda trn-sretel adapt-sre18-unlab' 'splda trn-sretel adapt-sre18eval+unlab')

be_names_caldev=(lda150_splday150_adapt_v3eval_a1_mu1B0.1W0.6_a2_M500_mu1B0.4W0.1_sre_tel)
be_tags_caldev=('splda coral trn-sretel+sre18eval adapt-sre18eval+unlab')

be_names_caleval=(lda150_splday125_adapt_v2dev_a1_mu1B0W0.5_a2_M500_mu1B0.75W0.6_sre_tel)
be_tags_caleval=('splda trn-sretel adapt-sre18dev+unlab')

. parse_options.sh || exit 1;
. $config_file

score_dir0=exp/scores/$nnet_name/

echo ""

#Tel table
args="--print_header true"
plda_dirs=(plda plda_snorm)
plda_tags=("" "s-norm")
nc=${#plda_dirs[*]}


nbe=${#be_names_caldev[*]}
for((i=0;i<$nbe;i++))
do
    for((j=0;j<$nc;j++))
    do
	d=${plda_dirs[$j]}
	be=${be_names_caldev[$i]}
	dir_i=$score_dir0/$be/${d}_cal_v1dev
	local/make_table_line_tel.sh $args "${nnet_name} ${be_tags_caldev[$i]} ${plda_tags[$j]} cal-sre18dev" $dir_i
	args=""
    done
done
exit
nbe=${#be_names_caleval[*]}
for((i=0;i<$nbe;i++))
do
    for((j=0;j<$nc;j++))
    do
	d=${plda_dirs[$j]}
	be=${be_names_caleval[$i]}
	dir_i=$score_dir0/$be/${d}_cal_v1eval
	local/make_table_line_tel.sh $args "${nnet_name} ${be_tags_caleval[$i]} ${plda_tags[$j]} cal-sre18eval" $dir_i
	args=""
    done
done


