#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

diar_name=diar1a
net_name=1a

tel_lda_dim=150
vid_lda_dim=200
tel_ncoh=400
vid_ncoh=500
vast_ncoh=120

w_mu1=1
w_B1=0.75
w_W1=0.75
w_mu2=1
w_B2=0.6
w_W2=0.6
num_spks=975

plda_tel_y_dim=125
plda_tel_z_dim=150
plda_vid_y_dim=150
plda_vid_z_dim=200

stage=1

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors/$net_name

coh_vid_data=sitw_sre18_dev_vast_${diar_name}
coh_vast_data=sitw_sre18_dev_vast_${diar_name}
coh_tel_data=sre18_dev_unlabeled
plda_tel_data=sre_tel_combined
plda_tel_type=splda
plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt_v1_a1_mu${w_mu1}B${w_B1}W${w_W1}_a2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}

plda_vid_data=voxceleb_combined
plda_vid_type=splda
plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1

be_tel_name=lda${tel_lda_dim}_${plda_tel_label}_${plda_tel_data}
be_vid_name=lda${vid_lda_dim}_${plda_vid_label}_${plda_vid_data}
be_tel_dir=exp/be/$net_name/$be_tel_name
be_vid_dir=exp/be/$net_name/$be_vid_name

score_dir=exp/scores/$net_name/tel_${be_tel_name}_vid_${be_vid_name}
score_plda_dir=$score_dir/plda
score_plda_diar_dir=${score_plda_dir}_${diar_name}
score_sre18_dir=$score_dir/sre18_plda_${diar_name}


# SITW Trials
sitw_dev_trials=data/sitw_dev_test/trials
sitw_eval_trials=data/sitw_eval_test/trials
sitw_dev_trials_diar=data/sitw_dev_test_${diar_name}/trials
sitw_eval_trials_diar=data/sitw_eval_test_${diar_name}/trials
sitw_conds=(core-core core-multi assist-core assist-multi)

# SRE18 trials
sre18_dev_trials_vast=data/sre18_dev_test_vast/trials
sre18_eval_trials_vast=data/sre18_eval_test_vast/trials
sre18_dev_trials_vast_diar=data/sre18_dev_test_vast_${diar_name}/trials
sre18_eval_trials_vast_diar=data/sre18_eval_test_vast_${diar_name}/trials

ldc_root=/export/corpora/LDC
sre18_dev_root=$ldc_root/LDC2018E46
sre18_eval_root=$ldc_root/LDC2018E51



if [ $stage -le 1 ];then

    #SITW
    echo "SITW dev with diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}

	steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
					$sitw_dev_trials/$cond_i.lst $sitw_dev_trials_diar/$cond_i.lst \
					data/sitw_dev_enroll/utt2spk \
					$xvector_dir/sitw_dev_${diar_name}/xvector.scp \
					data/sitw_dev_test_${diar_name}/utt2orig \
					$be_vid_dir/lda_lnorm_adapt.h5 \
					$be_vid_dir/plda.h5 \
					${score_plda_diar_dir}/sitw_dev_${cond_i}_scores &
    done

    
    echo "SITW eval with diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}

	steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
					$sitw_eval_trials/$cond_i.lst $sitw_eval_trials_diar/$cond_i.lst \
					data/sitw_eval_enroll/utt2spk \
					$xvector_dir/sitw_eval_${diar_name}/xvector.scp \
					data/sitw_eval_test_${diar_name}/utt2orig \
					$be_vid_dir/lda_lnorm_adapt.h5 \
					$be_vid_dir/plda.h5 \
					${score_plda_diar_dir}/sitw_eval_${cond_i}_scores &
    done
    wait

    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_diar_dir}
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_diar_dir}
    
fi


if [ $stage -le 2 ]; then

    #SRE18
    echo "SRE18 with diarization"
    
    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
    				    $sre18_dev_trials_vast $sre18_dev_trials_vast_diar \
    				    data/sre18_dev_enroll_vast/utt2spk \
    				    $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
				    data/sre18_dev_test_vast_${diar_name}/utt2orig \
    				    $be_vid_dir/lda_lnorm_adapt2.h5 \
    				    $be_vid_dir/plda.h5 \
    				    ${score_plda_diar_dir}/sre18_dev_vast_scores &

    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
    				    $sre18_eval_trials_vast $sre18_eval_trials_vast_diar \
    				    data/sre18_eval_enroll_vast/utt2spk \
    				    $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp \
				    data/sre18_eval_test_vast_${diar_name}/utt2orig \
    				    $be_vid_dir/lda_lnorm_adapt2.h5 \
    				    $be_vid_dir/plda.h5 \
    				    ${score_plda_diar_dir}/sre18_eval_vast_scores &
    wait
    
    local/score_sre18.sh $sre18_dev_root dev $score_plda_dir/sre18_dev_cmn2_scores ${score_plda_diar_dir}/sre18_dev_vast_scores ${score_sre18_dir}
    local/score_sre18.sh $sre18_eval_root eval $score_plda_dir/sre18_eval_cmn2_scores ${score_plda_diar_dir}/sre18_eval_vast_scores ${score_sre18_dir}

fi

if [ $stage -le 3 ];then
    local/calibrate_sitw_v1.sh --cmd "$train_cmd" $score_plda_diar_dir
    local/calibrate_sre18_v1.sh --cmd "$train_cmd" $score_plda_dir $score_plda_diar_dir
    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_diar_dir}_cal_v1
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_diar_dir}_cal_v1
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_dir}_cal_v1/sre18_dev_cmn2_scores ${score_plda_diar_dir}_cal_v1/sre18_dev_vast_scores ${score_sre18_dir}_cal_v1
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_dir}_cal_v1/sre18_eval_cmn2_scores ${score_plda_diar_dir}_cal_v1/sre18_eval_vast_scores ${score_sre18_dir}_cal_v1
    
fi

    
exit
