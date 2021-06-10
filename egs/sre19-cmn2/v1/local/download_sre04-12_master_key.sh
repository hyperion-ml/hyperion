#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads master key for sre04-12.
set -e 
key_name=master_key_sre04-12
master_key=$key_name/NIST_SRE_segments_key.v2.csv

# shareable link:
# https://drive.google.com/file/d/1znVYgrEuf9C0B1r7qNARB5E_h1EjKRGX/view?usp=sharing

#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1znVYgrEuf9C0B1r7qNARB5E_h1EjKRGX" -O $key_name.tar.gz

FILEID=1znVYgrEuf9C0B1r7qNARB5E_h1EjKRGX
FILENAME=$key_name.tar.gz
gdown https://drive.google.com/uc?id=$FILEID
tar xzvf $FILENAME

if [ ! -f $master_key ];then
    echo "master key wasn't dowloaded correctly"
    exit 1
fi

rm -f $key_name.tar.gz
