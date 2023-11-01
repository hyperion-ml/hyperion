#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

vocab_sizes=(
  # 5000
  2000
  1000
  500
)

dl_dir=$PWD/download

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. ./datapath.sh 
. $config_file


if [ $stage -le 1 ]; then
  echo "Stage 1: Download LM"
  mkdir -p $dl_dir/lm
  if [ ! -e $dl_dir/lm/.done ]; then
    ./local/download_lm.py --out-dir=$dl_dir/lm
    touch $dl_dir/lm/.done
  fi
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/librispeech-lexicon.txt |
    sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi
fi


if [ $stage -le 3 ]; then
  echo "Stage 3: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone/words.txt $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      echo "Generate data for BPE training"
      files=$(
        find "$librispeech_root/train-clean-100" -name "*.trans.txt"
        find "$librispeech_root/train-clean-360" -name "*.trans.txt"
        find "$librispeech_root/train-other-500" -name "*.trans.txt"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2-
      done > $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir

      echo "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi
  done
fi
