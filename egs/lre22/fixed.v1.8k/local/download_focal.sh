#!/bin/bash
# Copyright 2022 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads Niko Brummer's FoCal Multiclass

set -e 
tool=FoCal_MultiClass_V1
s_dir=focal_multiclass

# shareable link:
# https://drive.google.com/file/d/13rPUqS68NdEF5NB0vsL7bDEju5dhmmDZ/view?usp=sharing


wget --no-check-certificate "https://drive.google.com/uc?export=download&id=13rPUqS68NdEF5NB0vsL7bDEju5dhmmDZ" -O $tool.zip
unzip $tool.zip -d $s_dir

if [ ! -f $s_dir/v1.0/readme.txt ];then
    echo "the focal tool wasn't dowloaded correctly, download manually"
    exit 1
fi

rm -f $tool.zip



