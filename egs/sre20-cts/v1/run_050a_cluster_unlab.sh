#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

#spk det back-end
lda_dim=150
ncoh=500

w_mu=1
w_B=0.75
w_W=0.75
coral_mu=1
coral_T=1

plda_y_dim=125
plda_z_dim=150

plda_type=splda
plda_data=sre_tel
plda_adapt_data=realtel_noeng
coh_data=realtel_alllangs
# cal_set=sre16-9
cal_set=sre16-yue
ft=1
# ahc_thr=3 # p_tar ~ 0.01
ahc_thr=3 # p_tar ~ 0.05
min_sps=4

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
fi

cluster_label=ahc_v1_thr${ahc_thr}
plda_label=${plda_type}y${plda_y_dim}
be_name=lda${lda_dim}_${plda_label}_${plda_data}_v4_adapt_coral_mu${coral_mu}T${coral_T}_mu${w_mu}B${w_B}W${w_W}_${plda_adapt_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda_snorm_${coh_data}${ncoh}_cal_v1$cal_set/
cal_file=$score_plda_dir/cal_tel.h5

if [ $stage -le 1 ];then

    for name in $babel_datasets
    do
	steps_be/apply_ahc_v1.sh \
	    --cmd "$train_cmd" \
	    --plda-type $plda_type \
	    --ncoh $ncoh \
	    --cal-file $cal_file \
	    --class-prefix $name \
	    --thr $ahc_thr \
	    data/$name \
	    $xvector_dir/$name/xvector.scp \
	    $be_dir/lda_lnorm_adapt.h5 \
	    $be_dir/plda_adapt.h5 \
	    data/${name}_${cluster_label} &
    done
    wait
fi

if [ $stage -le 2 ];then

    for name in $lre17_datasets
    do
	steps_be/apply_ahc_v1.sh \
	    --cmd "$train_cmd" \
	    --plda-type $plda_type \
	    --ncoh $ncoh \
	    --cal-file $cal_file \
	    --class-prefix $name \
	    --thr $ahc_thr \
	    data/$name \
	    $xvector_dir/$name/xvector.scp \
	    $be_dir/lda_lnorm_adapt.h5 \
	    $be_dir/plda_adapt.h5 \
	    data/${name}_${cluster_label} &
    done
    wait
fi

if [ $stage -le 3 ];then
    output_dir=data/babel_alllangs_$cluster_label
    mkdir -p $output_dir
    rm -f $output_dir/*
    for name in $babel_datasets
    do
	for f in vad.scp utt2lang utt2spk wav.scp utt2num_frames utt2dur
	do
	    if [ -f data/${name}_${cluster_label}/$f ];then
		cat data/${name}_${cluster_label}/$f >> $output_dir/$f
	    fi
	done
    done
    utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt
    rm -rf ${output_dir}_minsps${min_sps}
    cp -r ${output_dir} ${output_dir}_minsps${min_sps}
    hyp_utils/remove_spk_few_utts_nosort.sh --min-num-utts $min_sps ${output_dir}_minsps${min_sps}

	# cat data/${name}_${cluster_label}/spk2utt >> $output_dir/spk2utt
    # 	cat data/${name}_${cluster_label}/wav.scp >> $output_dir/wav.scp
    # done
    # utils/spk2utt_to_utt2spk.pl $output_dir/spk2utt > $output_dir/utt2spk
    # rm -rf ${output_dir}_minsps${min_sps}
    # cp -r ${output_dir} ${output_dir}_minsps${min_sps}
    # awk -v min_sps=$min_sps 'NF > min_sps { print $0 }' $output_dir/spk2utt \
    # 	> ${output_dir}_minsps${min_sps}/spk2utt
    # utils/spk2utt_to_utt2spk.pl ${output_dir}_minsps${min_sps}/spk2utt \
    # 				> ${output_dir}_minsps${min_sps}/utt2spk
    # awk -v futts=${output_dir}_minsps${min_sps}/utt2spk \
    # 	-f local/filter_utts.awk ${output_dir}/wav.scp > \
    # 	${output_dir}_minsps${min_sps}/wav.scp
fi


if [ $stage -le 4 ];then
    output_dir=data/lre17_alllangs_$cluster_label
    mkdir -p $output_dir
    rm -f $output_dir/*
    for name in $lre17_datasets
    do
	for f in vad.scp utt2lang utt2spk wav.scp utt2num_frames utt2dur
	do
	    if [ -f data/${name}_${cluster_label}/$f ];then
		cat data/${name}_${cluster_label}/$f >> $output_dir/$f
	    fi
	done
    done
    utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt
    rm -rf ${output_dir}_minsps${min_sps}
    cp -r ${output_dir} ${output_dir}_minsps${min_sps}
    hyp_utils/remove_spk_few_utts_nosort.sh --min-num-utts $min_sps ${output_dir}_minsps${min_sps}
    # awk -v min_sps=$min_sps 'NF > min_sps { print $0 }' $output_dir/spk2utt \
    # 	> ${output_dir}_minsps${min_sps}/spk2utt
    # utils/spk2utt_to_utt2spk.pl ${output_dir}_minsps${min_sps}/spk2utt \
    # 				> ${output_dir}_minsps${min_sps}/utt2spk
    # awk -v futts=${output_dir}_minsps${min_sps}/utt2spk \
    # 	-f local/filter_utts.awk ${output_dir}/wav.scp > \
    # 	${output_dir}_minsps${min_sps}/wav.scp
fi

if [ $stage -le 5 ];then
    output_dir=data/babel_lre17_alllangs_${cluster_label}_minsps$min_sps
    mkdir -p $output_dir
    for file in utt2spk spk2utt 
    do
	for name in lre17_alllangs_${cluster_label}_minsps$min_sps \
				   babel_alllangs_${cluster_label}_minsps$min_sps
	do
	    cat data/$name/$file
	done > $output_dir/$file
    done
    mkdir -p $xvector_dir/babel_lre17_alllangs
    cat $xvector_dir/lre17_alllangs/xvector.scp \
	$xvector_dir/babel_alllangs/xvector.scp \
	> $xvector_dir/babel_lre17_alllangs/xvector.scp
fi




exit
