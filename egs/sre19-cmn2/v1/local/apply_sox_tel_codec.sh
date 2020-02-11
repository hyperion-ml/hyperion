#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 2 ]; then
    echo "Usage: $0 <data-in> <data-out>"
    exit 1
fi


data_in=$1
data_out=$2

mkdir -p $data_out
cp -r $data_in/* $data_out

awk ' { 
if (NR % 2 == 0){
   codec="amr-nb"
}
else{
   codec="gsm"
}
f=match($NF,/|$/);
if (f!=0){ 
print $0 " sox -t wav - -r 8000 -t "codec" - | sox -t "codec" - -t wav -e signed-integer -b 16 - |"
}
else{
print $1, "sox $2 -r 8000 -t "codec" - | sox -t "codec" - -t wav -e signed-integer -b 16 - |"
}
}' $data_in/wav.scp > $data_out/wav.scp

utils/fix_data_dir.sh $data_out

