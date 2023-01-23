#!/bin/bash


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
stop_stage=4
config_file=default_config.sh

. parse_options.sh || exit 1;
. ./datapath.sh 
. $config_file


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Dump transcripts for LM training"
  mkdir -p data/lm
  gunzip -c data/${language}/cv-${language}_supervisions_train.jsonl.gz \
    | jq '.text' \
    | sed 's:"::g' \
    > data/lm/${language}_transcript_words.txt
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/${language}_lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    # Add special words to words.txt
    echo "<eps> 0" > $lang_dir/words.txt
    echo "!SIL 1" >> $lang_dir/words.txt
    echo "<UNK> 2" >> $lang_dir/words.txt

    # Add regular words to words.txt
    gunzip -c data/${language}/cv-${language}_supervisions_train.jsonl.gz \
      | jq '.text' \
      | sed 's:"::g' \
      | sed 's: :\n:g' \
      | sort \
      | uniq \
      | sed '/^$/d' \
      | awk '{print $0,NR+2}' \
      >> $lang_dir/words.txt

    # Add remaining special word symbols expected by LM scripts.
    num_words=$(cat $lang_dir/words.txt | wc -l)
    echo "<s> ${num_words}" >> $lang_dir/words.txt
    num_words=$(cat $lang_dir/words.txt | wc -l)
    echo "</s> ${num_words}" >> $lang_dir/words.txt
    num_words=$(cat $lang_dir/words.txt | wc -l)
    echo "#0 ${num_words}" >> $lang_dir/words.txt

    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --vocab-size $vocab_size \
      --transcript data/lm/${language}_transcript_words.txt

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir
    fi
  done
fi

# if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
#   echo "Stage 3: Train LM"
#   lm_dir=data/lm

#   if [ ! -f $lm_dir/G.arpa ]; then
#     ./shared/make_kn_lm.py \
#       -ngram-order 3 \
#       -text $lm_dir/transcript_words.txt \
#       -lm $lm_dir/G.arpa
#   fi

#   if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
#     python3 -m kaldilm \
#       --read-symbol-table="data/lang_phone/words.txt" \
#       --disambig-symbol='#0' \
#       --max-order=3 \
#       $lm_dir/G.arpa > $lm_dir/G_3_gram.fst.txt
#   fi
# fi

# if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
#   echo "Stage 4: Compile HLG"
#   ./local/compile_hlg.py --lang-dir data/lang_phone

#   for vocab_size in ${vocab_sizes[@]}; do
#     lang_dir=data/lang_bpe_${vocab_size}
#     ./local/compile_hlg.py --lang-dir $lang_dir
#   done
# fi