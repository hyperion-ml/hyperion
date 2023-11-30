#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
nnet_stage=1
config_file=default_config.sh
use_gpu=false
do_tsne=false
split_dev=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd --mem 4G"
else
    xvec_cmd="$train_cmd --mem 12G"
fi

if [ $nnet_stages -lt $nnet_stage ];then
    nnet_stage=$nnet_stages
fi

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

xvector_dir=exp/xvectors/$nnet_name

# if [ $stage -le 1 ]; then
#     # Extract xvectors for training 
#   for name in lre17_proc_audio_no_sil \
# 		voxlingua107_codecs_proc_audio_no_sil \
# 		babel_sre_proc_audio_no_sil \
# 		cv_codecs_proc_audio_no_sil \
# 		others_afr_proc_audio_no_sil
#     do
#       steps_xvec/extract_xvectors_from_wav.sh \
# 	--cmd "$xvec_cmd" --nj 100 ${xvec_args} \
# 	--use-bin-vad false \
# 	--random-utt-length true --min-utt-length 300 --max-utt-length 3000 \
# 	--feat-config $feat_config \
#     	$nnet data/${name} \
#     	$xvector_dir/${name}
#     done
# fi

if [ $stage -le 2 ]; then
    # Extract xvectors for training 
    for name in lre22_dev
    do
	steps_xvec/extract_xvectors_from_wav.sh \
	    --cmd "$xvec_cmd" --nj 100 ${xvec_args} \
	    --use-bin-vad true --num-augs 10 --aug-config conf/reverb_noise_aug.yaml \
	    --random-utt-length true --min-utt-length 300 --max-utt-length 3000 \
	    --feat-config $feat_config \
    	    $nnet data/${name} \
    	    $xvector_dir/${name}_aug \
	    data/${name}_aug
    done
fi


if [ $stage -le 3 ]; then
    # Extracts x-vectors for dev and eval
    for name in lre22_dev lre22_eval
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_from_wav.sh \
	    --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi

if [ $stage -le 4 ]; then
    for name in lre22_dev
    do
	if [ "$do_tsne" == "true" ] || [ "$split_dev" == "true" ];then
	    $train_cmd \
		$xvector_dir/$name/tsne/tsne.log \
		hyp_utils/conda_env.sh \
		plot_embedding_tsne.py \
		--train-list data/$name/utt2lang \
		--train-v-file scp:$xvector_dir/$name/xvector.scp \
		--output-dir $xvector_dir/$name/tsne \
		--pca-var-r 0.975 \
		--lnorm \
		--prob-plot 1. \
		--tsne.metric cosine \
		--tsne.early-exaggeration 12 --tsne.perplexity 30

	    $train_cmd \
		$xvector_dir/$name/tsne_per_class/tsne.log \
		hyp_utils/conda_env.sh \
		plot_embedding_tsne_per_class.py \
		--train-list data/$name/utt2lang \
		--train-v-file scp:$xvector_dir/$name/xvector.scp \
		--output-dir $xvector_dir/$name/tsne_per_class \
		--pca-var-r 0.975 \
		--lnorm \
		--prob-plot 1. \
		--tsne.metric cosine \
		--tsne.early-exaggeration 12 --tsne.perplexity 30 \
		--do-ahc --cluster-tsne --ahc-thr -5

	    if [ "$split_dev" == "true" ];then
		hyp_utils/conda_env.sh \
		    local/split_dev.py \
		    --segs-file $xvector_dir/$name/tsne_per_class/segments.csv \
		    --output-dir ./resources/dev_splits \
		    --num-folds 2

		# delete the split data dirs so they are regenerated later
		rm -rf data/lre22_dev_p{1,2}

	    fi
	fi
    done
fi

if [ $stage -le 5 ]; then
    if [ ! -d data/lre22_dev_p1 ];then
	awk -F "," '$1!="id" { print $1}' \
	    ./resources/dev_splits/fold_0/train_segments.csv \
	    > p1.lst
	awk -F "," '$1!="id" { print $1}' \
	    ./resources/dev_splits/fold_0/test_segments.csv \
	    > p2.lst
	
	for p in p1 p2
	do
	    utils/subset_data_dir.sh \
		--utt-list $p.lst \
		data/lre22_dev data/lre22_dev_$p
	done
    fi
fi

if [ $stage -le 6 ]; then
    if [ -d data/lre22_dev_aug ] && [ ! -d data/lre22_dev_aug_p1 ];then
	awk -v fsegs=./resources/dev_splits/fold_0/train_segments.csv '
BEGIN{FS=",";
getline;
while(getline < fsegs)
{
   segs[$1]
}
FS=" ";
}
{ if($2 in segs){ print $1}}' data/lre22_dev_aug/augm2clean \
    > p1.lst

	awk -v fsegs=./resources/dev_splits/fold_0/test_segments.csv '
BEGIN{FS=",";
getline;
while(getline < fsegs)
{
   segs[$1]=1;
}
FS=" ";
}
{ if($2 in segs){ print $1}}' data/lre22_dev_aug/augm2clean \
    > p2.lst

	for p in p1 p2
	do
	    utils/subset_data_dir.sh \
		--utt-list $p.lst \
		data/lre22_dev_aug data/lre22_dev_aug_$p
	done
    fi
fi

if [ $stage -le 7 ];then
    if [ -f $xvector_dir/lre22_dev_aug/xvector.scp ];then
	mkdir -p $xvector_dir/lre22_dev_aug_clean
	cat $xvector_dir/lre22_dev/xvector.scp \
	    $xvector_dir/lre22_dev_aug/xvector.scp \
	    > $xvector_dir/lre22_dev_aug_clean/xvector.scp

	for p in "" _p1 _p2
	do
	    if [ ! -d data/lre22_dev_aug_clean$p ]; then
		utils/combine_data.sh \
		    data/lre22_dev_aug_clean$p \
		    data/lre22_dev$p \
		    data/lre22_dev_aug$p
	    fi
	done
    fi
fi

exit
