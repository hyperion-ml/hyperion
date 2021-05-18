#!/bin/bash
# Copyright 2019  Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [  $# != 2 ]; then
    echo "$0 <db-path> <output_path>"
    exit 1
fi

input_path=$1
output_path=$2

echo "$0 making chime5 spkdet enroll"
enroll_segm=$input_path/dgr_chime5_enroll_segments
data_out=$output_path/chime5_spkdet_enroll
audio_dir=$input_path/enroll/BIN.SUM
mkdir -p $data_out

awk '{ split($2,f,"_"); s=f[1]; print $2,s }' \
    $enroll_segm | sort -u > $data_out/utt2spk
utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt

awk '{ print $1,"'$audio_dir'/"$1".wav" }' \
    $data_out/utt2spk > $data_out/wav.scp

awk '{ print $0 }' $enroll_segm > $data_out/diarization_segments
awk '{ split($2,f,"_"); spk=f[1]; print "SPEAKER",$2,"1",$3,$4-$3,"<NA> <NA>",spk,"<NA> <NA>"}' $data_out/diarization_segments > $data_out/rttm

utils/fix_data_dir.sh $data_out

models=$data_out/spk2utt

echo "$0 making chime5 spkdet test"

test_segm=$input_path/dgr_chime5_test_segments
data_out=$output_path/chime5_spkdet_test
audio_dir=$input_path/test
mkdir -p $data_out
awk '!/P20_S07_.*_0024/ { 
    split($2,f,"_"); s=f[1]; print $2,s }' \
    $test_segm | sort -u > $data_out/utt2spk
utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt

awk '{ split($1,f,"_"); dir=f[3]; 
       print $1,"'$audio_dir'/"dir"/"$1".wav" }' \
    $data_out/utt2spk > $data_out/wav.scp

awk '!/P20_S07_.*_0024/ { print $0 }' $test_segm > $data_out/diarization_segments

awk -v fm=$models '
function merge_sessions(sess_id) {
   #merge sessions with same spks
   sub(/04/,"03", sess_id);
   sub(/06/,"05", sess_id);
   sub(/17/,"07", sess_id);
   sub(/16/,"08", sess_id);
   sub(/13/,"12", sess_id);
   sub(/20/,"19", sess_id);
   sub(/22/,"18", sess_id);
   sub(/24/,"23", sess_id);
   return sess_id
}
BEGIN{
   n_models=0;
   while(getline < fm)
   {
       split($2,f,"_"); sess=f[2];
       sess=merge_sessions(sess);       
       v_mod[n_models]=$1;
       v_sess[n_models]=sess;
       n_models++;
   }
}
{
   split($1,f,"_"); spk=f[1]; sess=f[2];
   sess=merge_sessions(sess);
   for(i=0;i<n_models;i++)
   {
        if(spk==v_mod[i]){ 
            print v_mod[i],$1,"target";
        }
        else{ 
            if (sess!=v_sess[i])
                print v_mod[i],$1,"nontarget";
        }
   }
}' $data_out/utt2spk | sort -k1,2 > $data_out/trials

for cond in BIN.SUM U01.CH1 U02.CH1 U04.CH1 U06.CH1
do
    awk '$3=="nontarget" || $2 ~ /'$cond'/ ' $data_out/trials > $data_out/trials_$cond
done

utils/fix_data_dir.sh $data_out
