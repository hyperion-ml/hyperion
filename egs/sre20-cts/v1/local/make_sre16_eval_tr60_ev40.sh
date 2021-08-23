#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE16_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
meta=$input_path/metadata
call2lang=$meta/calls.tsv
call2spk=$meta/call_sides.tsv
spk2gender=$meta/subjects.tsv
enroll_file=$docs/sre16_eval_enrollment.tsv
segm_file=$docs/sre16_eval_segment_key.tsv
trial_file=$docs/sre16_eval_trials.tsv
key_file=$docs/sre16_eval_trial_key.tsv

tel_up=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
fi

#Dev CMN2 Cantonese and Tagalog
for lang in yue tgl
do
    output_dir=$output_path/sre16_eval_tr60_${lang}
    trn_list=$output_dir/spk_trn60
    ev_list=$output_dir/spk_ev40
    mkdir -p $output_dir
    awk -v c2l=$call2lang -v c2s=$call2spk -v s2g=$spk2gender -v l=$lang -F "\t" 'BEGIN{ 
while(getline < c2l)
{
     if($2 == l){ calls[$1]=1 }
}
while(getline < c2s) { spk[$1]=$3 }
while(getline < s2g) { gender[$1]=tolower($2) }
}
{ if($2 in calls) { s=spk[$2]; print $1, s, gender[s] }}' $segm_file > $output_dir/table

    #split spks into 60/40
    awk 'BEGIN{n=0} 
    { if(!($2 in spk)){ spk[$2]=1; n++} } 
     END{ n_trn=int(0.6*n+0.5); count=0; 
          for(v in spk){ 
              if(count < n_trn){ 
                    print v > "'$trn_list'";
              }
              else {
                    print v > "'$ev_list'";
              }
              count++;  
                
          }
     }
' $output_dir/table

    awk -v fspk=$trn_list 'BEGIN{ while(getline < fspk){spks[$1]=1} }
    { if($2 in spks){ print $0}}' $output_dir/table > $output_dir/table_tr60

    
    awk '{ print $2"-"$1,$2}' $output_dir/table_tr60 | sort -k1,1 > $output_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt
    awk '{ print $2,$3}' $output_dir/table_tr60 | sort -k1,1 -u > $output_dir/spk2gender
    awk -v lang=$lang '{ print $1,toupper(lang)}' $output_dir/utt2spk > $output_dir/utt2lang
    
    find -L $input_path -name "*.sph" > $output_dir/wav.scp.tmp    

    awk -v fwav=$output_dir/wav.scp.tmp 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   sub(/\.sph$/,"",bn);
   wav[bn]=$1;
}
}
{  print $2"-"$1,"sph2pipe -f wav -p -c 1 "wav[$1]" |'"$tel_up"'"}' $output_dir/table_tr60 | \
    sort -k1,1 > $output_dir/wav.scp

    utils/fix_data_dir.sh $output_dir
    utils/validate_data_dir.sh --no-text --no-feats $output_dir

    # Create Eval dirs with 40% of spks
    #Enrollment CMN2
    enroll_dir=$output_path/sre16_eval40_${lang}_enroll
    mkdir -p $enroll_dir
    awk -v fspk=$ev_list 'BEGIN{ while(getline < fspk){ spks[$1]=1} }
    { if($2 in spks){ print $0}}' $output_dir/table > $enroll_dir/table_ev40

    awk -v fseg=$enroll_dir/table_ev40 'BEGIN{
while(getline < fseg)
{ 
    files[$1]=1;
}
files["segment"]=1;
}
{ if($2 in files) { print $0 }}' $enroll_file > $enroll_dir/enr40.tsv
    awk -v lang=$lang '{ print $1,toupper(lang)}' $enroll_dir/utt2spk > $enroll_dir/utt2lang

    find -L $input_path -name "*.sph" > $enroll_dir/wav.scp.tmp    
    awk -v fwav=$enroll_dir/wav.scp.tmp 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   sub(/\.sph$/,"",bn);
   wav[bn]=$1;
}
}
!/modelid/ {  print $1"-"$2,"sph2pipe -f wav -p -c 1 "wav[$2]" |'"$tel_up"'"}' $enroll_dir/enr40.tsv | \
    sort -k1,1 > $enroll_dir/wav.scp
    
    awk '!/modelid/ { print $1"-"$2,$1}' $enroll_dir/enr40.tsv | sort -k1,1 > $enroll_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt
    
    utils/fix_data_dir.sh $enroll_dir
    utils/validate_data_dir.sh --no-text --no-feats $enroll_dir

    #Test set CMN2
    test_dir=$output_path/sre16_eval40_${lang}_test
    mkdir -p $test_dir
    awk -v fseg=$enroll_dir/table_ev40 -v fenr=$enroll_dir/enr40.tsv 'BEGIN{
files["segment"]=1;
while(getline < fseg)
{
    files[$1]=1;
}
enr["modelid"] = 1
while(getline < fenr)
{ 
    enr[$1]=1;
}

}
{ if(($1 in enr) && ($2 in files)) { print $0 }}' $key_file > $test_dir/trial_key.tsv

    key_file40=$test_dir/trial_key.tsv
    awk 'BEGIN{ OFS="\t"} { print $1,$2,$3}' $key_file40 > $test_dir/trials.tsv
    awk '!/modelid/ { print $1,$2,$4 }' $key_file40 > $test_dir/trials
    
    find -L $input_path -name "*.sph" > $test_dir/wav.scp.tmp    
    awk -v fwav=$test_dir/wav.scp.tmp 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   sub(/\.sph$/,"",bn);
   wav[bn]=$1;
}
}
!/modelid/ {  print $2,"sph2pipe -f wav -p -c 1 "wav[$2]" |'"$tel_up"'"}' $test_dir/trials.tsv | \
    sort -u -k1,1 > $test_dir/wav.scp

    awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
    cp $test_dir/utt2spk $test_dir/spk2utt
    awk -v lang=$lang '{ print $1,toupper(lang)}' $test_dir/utt2spk > $test_dir/utt2lang
    
    utils/fix_data_dir.sh $test_dir
    utils/validate_data_dir.sh --no-text --no-feats $test_dir

done
