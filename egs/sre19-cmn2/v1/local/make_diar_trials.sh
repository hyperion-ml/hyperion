#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

if [ $# -ne 3 ]; then
  echo "Usage: $0 <utt_id-to-cluster_id> <trial-list> <trial-out>"
  exit 1;
fi


orig2utt=$1
trial_in=$2
trial_out=$3

awk -v f=$orig2utt 'BEGIN{
     while(getline < f)
     {
        diar[$1]=$0;
     }
}
{
   v=diar[$2];
   nd=split(v,d," ");
   for(i=2;i<=nd;i++)
   {
      $2=d[i];
      print $0;
   } 

}' $trial_in > $trial_out

