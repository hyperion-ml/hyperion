#!/bin/bash

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

