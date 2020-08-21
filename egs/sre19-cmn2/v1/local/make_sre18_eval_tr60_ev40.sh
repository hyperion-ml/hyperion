#!/bin/bash

# Copyright 2018 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE18_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
enroll_file=$docs/sre18_eval_enrollment.tsv
enroll_diar_file=$docs/sre18_eval_enrollment_diarization.tsv
segm_file=$docs/sre18_eval_segment_key.tsv
trial_file=$docs/sre18_eval_trials.tsv
key_file=$docs/sre18_eval_trial_key.tsv

tel_up=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
elif [ $fs -eq 8 ];then
    tel_up=""
fi

#60% of the speakers will be used for adaptation 40% for evaluation
# Make the adaptation directory

output_dir=$output_path/sre18_eval_cmn2_tr60
trn_list=$output_dir/spk_trn60
ev_list=$output_dir/spk_ev40
mkdir -p $output_dir
#split spks into 60/40
awk 'BEGIN{n=0} 
     $7=="cmn2" && $4 != "unlabeled" { if(!($2 in spk)){ spk[$2]=1; n++} } 
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
' $segm_file

awk -v fspk=$trn_list 'BEGIN{
   while(getline < fspk)
   { 
      spk[$0]=1;
   }
}
$7=="cmn2" && $4 != "unlabeled" { if($2 in spk){ print $2"-"$1,$2}}' $segm_file | sort -k1,1 > $output_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt

find $input_path -name "*.sph" > $output_dir/wav.scp.tmp

awk -v fwav=$output_dir/wav.scp.tmp -v fspk=$trn_list 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   wav[bn]=$1;
}
while(getline < fspk)
{ 
    spk[$0]=1;
}
}
$7=="cmn2" && $4 != "unlabeled" {  if($2 in spk) { print $2"-"$1,"sph2pipe -f wav -p -c 1 "wav[$1]" |'"$tel_up"'"}}' $segm_file | \
    sort -k1,1 > $output_dir/wav.scp

rm -f $output_dir/wav.scp.tmp

awk -v sf=$segm_file 'BEGIN{
while(getline < sf)
{
 gender[$1]=substr($3,1,1)
}
}
{ sub(/^[^-]*-/,"",$2); print $1,gender[$2] } ' $output_dir/spk2utt > $output_dir/spk2gender

utils/fix_data_dir.sh $output_dir
utils/validate_data_dir.sh --no-text --no-feats $output_dir


# Create Eval dirs with 40% of spks

#Enrollment CMN2
enroll_dir=$output_path/sre18_eval40_enroll_cmn2
mkdir -p $enroll_dir
awk -v fseg=$segm_file -v fspk=$ev_list 'BEGIN{
while(getline < fspk)
{ 
    spk[$0]=1;
}
files["segmentid"]=1;
while(getline < fseg)
{
    if ($2 in spk){ files[$1]=1;}
}
}
{ if($2 in files) { print $0 }}' $enroll_file > $enroll_dir/enr40.tsv

awk '/\.sph/ { print $1"-"$2,"sph2pipe -f wav -p -c 1 '$input_path'/data/enrollment/"$2" |'"$tel_up"'"}' $enroll_dir/enr40.tsv | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '!/modelid/ && /\.sph/ { print $1"-"$2,$1}' $enroll_dir/enr40.tsv | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir

#Test set CMN2
test_dir=$output_path/sre18_eval40_test_cmn2
mkdir -p $test_dir
awk -v fseg=$segm_file -v fspk=$ev_list -v fenr=$enroll_dir/enr40.tsv 'BEGIN{
while(getline < fspk)
{ 
    spk[$1]=1;
}
files["segmentid"]=1;
while(getline < fseg)
{
    if ($2 in spk){ files[$1]=1;}
}
enr["modelid"] = 1
while(getline < fenr)
{ 
    enr[$1]=1;
}

}
{ if(($1 in enr) && ($2 in files)) { print $0 }}' $key_file > $test_dir/trial_key.tsv

key_file=$test_dir/trial_key.tsv
awk 'BEGIN{ OFS="\t"} { print $1,$2,$3}' $key_file > $test_dir/trials.tsv

awk '/\.sph/ { print $2,"sph2pipe -f wav -p -c 1 '$input_path'/data/test/"$2" |'"$tel_up"'"}' $key_file | \
    sort -u -k1,1 > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

#awk '!/modelid/ && /\.sph/ { print $1,$2 }' $ndx_file > $test_dir/trials
awk '!/modelid/ && $9=="cmn2" { print $1,$2,$4 }' $key_file > $test_dir/trials
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" && ($6=="Y" || $4=="nontarget") { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn_samephn
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" && $6=="N" { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn_diffphn
awk '!/modelid/ && $9=="cmn2" && $8=="voip" { print $1,$2,$4 }' $key_file > $test_dir/trials_voip

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir

