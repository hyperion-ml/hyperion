#!/bin/bash

data_dir=$1
vad_dir=$2

mkdir -p $vad_dir

feat-to-len scp:$data_dir/feats.scp ark,t:$data_dir/utt2num_frames

name=$(basename $data_dir)

cp $data_dir/vad.scp $data_dir/vad.scp.old



copy-vector scp:$data_dir/vad.scp.old ark,t:- | \
    awk -v f2l=$data_dir/utt2num_frames 'BEGIN{
while(getline < f2l)
{
   l[$1]=$2
}
}
{ 
num_frames_in=NF-3;
num_frames_out=l[$1]
if (num_frames_in>num_frames_out)
{
   $(num_frames_out+3)="]";
   for(i=num_frames_out+4;i<=num_frames_in+3;i++){ $i=""}
}
else if (num_frames_in<num_frames_out)
{
  for(i=num_frames_in+3;i<num_frames_out+3;i++){ $i="0"};
  $(num_frames_out+3)="]";
};
print $0
}' | copy-vector ark,t:- ark,scp:$vad_dir/vad_${name}.ark,$data_dir/vad.scp



