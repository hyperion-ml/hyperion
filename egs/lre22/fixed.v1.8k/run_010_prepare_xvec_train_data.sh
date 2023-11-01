#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 1 ]; then
  # This script preprocess audio for x-vector training
  for name in voxlingua107_codecs \
		lre17_train \
  		lre17_{dev,eval}_{cts,afv,afv_codecs}
  do
    steps_xvec/preprocess_audios_for_nnet_train.sh \
      --nj 40 --cmd "$train_cmd" \
      --storage_name lre22-fixed-v1.8k-$(date +'%m_%d_%H_%M') --use-bin-vad true \
      data/${name} data/${name}_proc_audio_no_sil exp/${name}_proc_audio_no_sil
    utils/fix_data_dir.sh data/${name}_proc_audio_no_sil
  done
fi

if [ $stage -le 2 ];then
  utils/combine_data.sh \
    data/lre17_proc_audio_no_sil \
    data/lre17_train_proc_audio_no_sil \
    data/lre17_{dev,eval}_{cts,afv,afv_codecs}_proc_audio_no_sil
fi

if [ $stage -le 3 ]; then
  # Now, we remove files with less than 3s
  hyp_utils/remove_short_audios.sh --min-len 3 data/voxlingua107_codecs_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/lre17_proc_audio_no_sil
fi

if [ $stage -le 4 ];then
  # merge voxlingua and lre17
  utils/combine_data.sh \
    data/voxlingua107_lre17_proc_audio_no_sil \
    data/voxlingua107_codecs_proc_audio_no_sil \
    data/lre17_proc_audio_no_sil
fi

if [ $stage -le 5 ]; then
  for name in lre17_proc_audio_no_sil  voxlingua107_lre17_proc_audio_no_sil
  do
    hyp_utils/conda_env.sh \
      local/split_segments_train_val.py \
      --segments-file data/$name/utt2lang \
      --recordings-file data/$name/wav.scp \
      --durations-file data/$name/utt2dur \
      --val-percent 2. \
      --output-dir data/$name/train_val_split
  done
fi

if [ $stage -le 6 ]; then
  for name in voxlingua107_lre17_proc_audio_no_sil
  do
    hyp_utils/conda_env.sh \
      local/split_segments_train_val.py \
      --segments-file data/$name/utt2lang \
      --recordings-file data/$name/wav.scp \
      --durations-file data/$name/utt2dur \
      --remove-langs en-en es-es ar-ar pt-pt \
      --val-percent 2. \
      --ara-ary-seg-file resources/lre17_ara-ary/segs_ara-ary.csv \
      --output-dir data/$name/train_val_split_noary
  done
  mkdir data/voxlingua107_lre17_noary_proc_audio_no_sil
  cd data/voxlingua107_lre17_noary_proc_audio_no_sil
  ln -s ../voxlingua107_lre17_proc_audio_no_sil/wav.scp
  ln -s ../voxlingua107_lre17_proc_audio_no_sil/train_val_split_noary train_val_split
  cd -
  
fi

if [ $stage -le 7 ]; then
  awk 'BEGIN{
adapt_langs_list="ara-acm ara-aeb ara-apc ara-arq ara-arz ara-ayl eng-gbr eng-usg por-brz zho-cmn zho-nan am-am sn-sn fra-mix haw-haw ia-ia ceb-ceb tl-tl sa-sa su-su te-te yo-yo sw-sw war-war km-km tr-tr gn-gn ha-ha ln-ln mg-mg";
nf=split(adapt_langs_list, f, " "); 
for(i=1;i<=nf;i++){ adapt_langs[f[i]]=1;};
FS=","; OFS=",";
getline; print $0;
}
{if ($1 in adapt_langs) { $3="1."} else{ $3="0.01"}; print $0}' \
      data/voxlingua107_lre17_noary_proc_audio_no_sil/train_val_split/class_file.csv > \
      data/voxlingua107_lre17_noary_proc_audio_no_sil/train_val_split/class_file_adapt_1.csv
fi
