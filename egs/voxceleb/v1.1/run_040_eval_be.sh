#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nnet_stage=1
config_file=default_config.sh


. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
elif [ $nnet_stage -eq 3 ];then
  nnet=$nnet_s3
  nnet_name=$nnet_s3_name
elif [ $nnet_stage -eq 4 ];then
  nnet=$nnet_s4
  nnet_name=$nnet_s4_name
elif [ $nnet_stage -eq 5 ];then
  nnet=$nnet_s5
  nnet_name=$nnet_s5_name
elif [ $nnet_stage -eq 6 ];then
  nnet=$nnet_s6
  nnet_name=$nnet_s6_name
fi

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
score_cosine_dir=exp/scores/$nnet_name/cosine
score_cosine_snorm_dir=exp/scores/$nnet_name/cosine_snorm
score_cosine_qmf_dir=exp/scores/$nnet_name/cosine_qmf


if [ "$do_plda" == "true" ];then
  if [ $stage -le 1 ]; then
    echo "Train PLDA on Voxceleb2"
    steps_be/train_be_v1.sh \
      --cmd "$train_cmd" \
      --lda_dim $lda_dim \
      --plda_type $plda_type \
      --y_dim $plda_y_dim --z_dim $plda_z_dim \
      $xvector_dir/$plda_data/xvector.scp \
      data/$plda_data \
      $be_dir
    
  fi
  
  
  if [ $stage -le 2 ];then
    echo "Eval Voxceleb 1 with LDA+CentWhiten+LNorm+PLDA"
    steps_be/eval_be_v1.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/voxceleb1_test/trials \
      data/voxceleb1_test/utt2model \
      $xvector_dir/voxceleb1_test/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/voxceleb1_scores

    $train_cmd --mem 10G --num-threads 6 $score_plda_dir/log/score_voxceleb1.log \
    	       local/score_voxceleb1.sh data/voxceleb1_test $score_plda_dir 
    
    for f in $(ls $score_plda_dir/*_results);
    do
      echo $f
      cat $f
      echo ""
    done
  fi
fi



if [ $stage -le 3 ];then

  echo "Eval Voxceleb 1 with Cosine scoring"
  steps_be/eval_be_cos.sh \
    --cmd "$train_cmd" \
    data/voxceleb1_test/trials \
    data/voxceleb1_test/utt2model \
    $xvector_dir/voxceleb1_test/xvector.scp \
    $score_cosine_dir/voxceleb1_scores

  $train_cmd --mem 12G --num-threads 6 $score_cosine_dir/log/score_voxceleb1.log \
	     local/score_voxceleb1.sh data/voxceleb1_test $score_cosine_dir 

  for f in $(ls $score_cosine_dir/*_results);
  do
    echo $f
    cat $f
    echo ""
  done

fi

if [ $stage -le 4 ] && [ "$do_voxsrc22" == "true" ];then

    echo "Eval voxsrc2 with Cosine scoring"
    steps_be/eval_be_cos.sh --cmd "$train_cmd" \
    	data/voxsrc22_dev/trials \
    	data/voxsrc22_dev/utt2model \
    	$xvector_dir/voxsrc22_dev/xvector.scp \
    	$score_cosine_dir/voxsrc22_dev_scores &

    # steps_be/eval_be_cos.sh --cmd "$train_cmd" \
    # 	data/voxsrc22_test/trials \
    # 	data/voxsrc22_test/utt2model \
    # 	$xvector_dir/voxsrc22_test/xvector.scp \
    # 	$score_cosine_dir/voxsrc22_test_scores

    wait
    $train_cmd --mem 10G --num-threads 1 $score_cosine_dir/log/score_voxsrc22_dev.log \
	local/score_voxsrc22_dev.sh data/voxsrc22_dev $score_cosine_dir 

    for f in $(ls $score_cosine_dir/voxsrc22_dev_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi


if [ "$do_snorm" == "true" ];then
  if [ $stage -le 5 ];then
    echo "Eval Voxceleb 1 with Cosine scoring + Adaptive SNorm"
    steps_be/eval_be_cos_snorm.sh \
      --cmd "$train_cmd --mem 22G" --coh-nbest 1000 \
      data/voxceleb1_test/trials \
      data/voxceleb1_test/utt2model \
      $xvector_dir/voxceleb1_test/xvector.scp \
      data/voxceleb2cat_train/utt2spk \
      $xvector_dir/voxceleb2cat_train/xvector.scp \
      $score_cosine_snorm_dir/voxceleb1_scores
    
    $train_cmd --mem 10G --num-threads 6 $score_cosine_snorm_dir/log/score_voxceleb1.log \
	       local/score_voxceleb1.sh data/voxceleb1_test $score_cosine_snorm_dir 
    
    for f in $(ls $score_cosine_snorm_dir/*_results);
    do
      echo $f
      cat $f
      echo ""
    done
  fi

  if [ $stage -le 6 ];then
    echo "Eval voxsrc2 with Cosine scoring"
    steps_be/eval_be_cos_snorm.sh \
      --cmd "$train_cmd --mem 20G" --coh-nbest 1000 \
      data/voxsrc22_dev/trials \
      data/voxsrc22_dev/utt2model \
      $xvector_dir/voxsrc22_dev/xvector.scp \
      data/voxceleb2cat_train/utt2spk \
      $xvector_dir/voxceleb2cat_train/xvector.scp \
      $score_cosine_snorm_dir/voxsrc22_dev_scores &
    
    # steps_be/eval_be_cos_snorm.sh --cmd "$train_cmd" \
    # 	data/voxsrc22_test/trials \
    # 	data/voxsrc22_test/utt2model \
    # 	$xvector_dir/voxsrc22_test/xvector.scp \
    #   data/voxceleb2cat_train/utt2spk \
    #	$xvector_dir/voxceleb2cat_train/xvector.scp \
    # 	$score_cosine_snorm_dir/voxsrc22_test_scores

    wait
    $train_cmd --mem 10G --num-threads 1 $score_cosine_snorm_dir/log/score_voxsrc22_dev.log \
	local/score_voxsrc22_dev.sh data/voxsrc22_dev $score_cosine_snorm_dir 

    for f in $(ls $score_cosine_snorm_dir/voxsrc22_dev_results);
    do
	echo $f
	cat $f
	echo ""
    done
  fi
fi


if [ "$do_qmf" == "true" ];then
  if [ $stage -le 7 ];then
    echo "Train QMF in Vox2"
    steps_be/train_be_cos_qmf.sh \
      --cmd "$train_cmd" --coh-nbest 1000 \
      data/voxceleb2cat_train/trials \
      data/voxceleb2cat_train/utt2model \
      $xvector_dir/voxceleb2cat_train/xvector.scp \
      $xvector_dir/voxceleb2cat_train/utt2num_frames \
      data/voxceleb2cat_train/snorm_utt2spk \
      $xvector_dir/voxceleb2cat_train/xvector.scp \
      $score_cosine_qmf_dir/voxceleb2_qmf_scores

  fi

  if [ $stage -le 8 ];then

    echo "Eval Voxceleb 1 with Cosine scoring"
    steps_be/eval_be_cos_qmf.sh \
      --cmd "$train_cmd --mem 20G" --coh-nbest 1000 \
      data/voxceleb1_test/trials \
      data/voxceleb1_test/utt2model \
      $xvector_dir/voxceleb1_test/xvector.scp \
      $xvector_dir/voxceleb1_test/utt2num_frames \
      data/voxceleb2cat_train/utt2spk \
      $xvector_dir/voxceleb2cat_train/xvector.scp \
      $score_cosine_qmf_dir/qmf.h5 \
      $score_cosine_qmf_dir/voxceleb1_scores
    
    $train_cmd --mem 10G --num-threads 6 $score_cosine_qmf_dir/log/score_voxceleb1.log \
	       local/score_voxceleb1.sh data/voxceleb1_test $score_cosine_qmf_dir 
    $train_cmd --mem 10G --num-threads 6 $score_cosine_qmf_dir/log/score_voxceleb1_snorm.log \
	       local/score_voxceleb1.sh data/voxceleb1_test $score_cosine_qmf_dir _snorm
    $train_cmd --mem 10G --num-threads 6 $score_cosine_qmf_dir/log/score_voxceleb1_qmf.log \
	       local/score_voxceleb1.sh data/voxceleb1_test $score_cosine_qmf_dir _qmf

    for f in $(ls $score_cosine_qmf_dir/voxceleb1{,_snorm,_qmf}_[oeh]_clean_results);
    do
      echo $f
      cat $f
      echo ""
    done

  fi

  if [ $stage -le 9 ];then
    echo "Eval voxsrc2 with Cosine scoring"
    steps_be/eval_be_cos_qmf.sh \
      --cmd "$train_cmd  --mem 20G" --coh-nbest 1000  \
      data/voxsrc22_dev/trials \
      data/voxsrc22_dev/utt2model \
      $xvector_dir/voxsrc22_dev/xvector.scp \
      $xvector_dir/voxsrc22_dev/utt2num_frames \
      data/voxceleb2cat_train/utt2spk \
      $xvector_dir/voxceleb2cat_train/xvector.scp \
      $score_cosine_qmf_dir/qmf.h5 \
      $score_cosine_qmf_dir/voxsrc22_dev_scores &

    # steps_be/eval_be_cos_qmf.sh --cmd "$train_cmd" \
    # 	data/voxsrc22_test/trials \
    # 	data/voxsrc22_test/utt2model \
    # 	$xvector_dir/voxsrc22_test/xvector.scp \
    #	$xvector_dir/voxsrc22_test/utt2num_frames \
    #	data/voxceleb2cat_train/utt2spk \
    #	$xvector_dir/voxceleb2cat_train/xvector.scp \
    #	$score_cosine_qmf_dir/qmf.h5 \
    # 	$score_cosine_qmf_dir/voxsrc22_test_scores

    wait
    $train_cmd --mem 10G --num-threads 1 $score_cosine_qmf_dir/log/score_voxsrc22_dev.log \
	local/score_voxsrc22_dev.sh data/voxsrc22_dev $score_cosine_qmf_dir 
    $train_cmd --mem 10G --num-threads 1 $score_cosine_qmf_dir/log/score_voxsrc22_dev_snorm.log \
	local/score_voxsrc22_dev.sh data/voxsrc22_dev $score_cosine_qmf_dir _snorm
    $train_cmd --mem 10G --num-threads 1 $score_cosine_qmf_dir/log/score_voxsrc22_dev_qmf.log \
	local/score_voxsrc22_dev.sh data/voxsrc22_dev $score_cosine_qmf_dir _qmf

    for f in $(ls $score_cosine_qmf_dir/voxsrc22_dev{,_snorm,_qmf}_results);
    do
	echo $f
	cat $f
	echo ""
    done
  fi

fi

if [ "$do_pca" != "true" ];then
  exit 0
fi


be_name=pca_r${pca_var_r}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_cosine_dir=exp/scores/$nnet_name/$be_name/cosine
score_cosine_snorm_dir=exp/scores/$nnet_name/$be_name/cosine_snorm
score_cosine_qmf_dir=exp/scores/$nnet_name/$be_name/cosine_qmf

be_dir=exp/be/$nnet_name/
score_be_dir=$score_dir/pca_r${pca_var_r}

if [ $stage -le 10 ]; then
  echo "Train projection on Voxceleb2"
  $train_cmd $be_dir/log/train_be.log \
	     hyp_utils/conda_env.sh \
	     steps_be/train_be_proj_v1.py \
	     --v-file scp:$xvector_dir/$plda_data/xvector.scp \
	     --train-list data/$plda_data/utt2spk \
	     --output-path $be_dir \
	     --pca.pca-var-r $pca_var_r

fi


if [ $stage -le 11 ];then

  echo "Eval Voxceleb 1 with Cosine scoring"
  steps_be/eval_be_cos.sh \
    --cmd "$train_cmd" \
    --preproc-file $be_dir/preproc.h5 \
    data/voxceleb1_test/trials \
    data/voxceleb1_test/utt2model \
    $xvector_dir/voxceleb1_test/xvector.scp \
    $score_cosine_dir/voxceleb1_scores

  $train_cmd --mem 10G --num-threads 6 $score_cosine_dir/log/score_voxceleb1.log \
	     local/score_voxceleb1.sh data/voxceleb1_test $score_cosine_dir 

  for f in $(ls $score_cosine_dir/*_results);
  do
    echo $f
    cat $f
    echo ""
  done

fi
