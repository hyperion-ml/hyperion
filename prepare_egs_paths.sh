#!/bin/bash
# Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

conda=""
kaldi=""
env=""

. hyp_utils/parse_options.sh || exit 1;

if [[ -z "$conda" && -z "$kaldi" && -z "$env" ]];then
    echo "Usage: $0 --conda <path-to-anaconda> --kaldi <path-to-kaldi> --env <conda-environment>"
    echo "e.g.: $0 --conda /home/janto/usr/local/anaconda3 --kaldi /export/b15/janto/kaldi/kaldi-villalba --env hyperion"
    exit 1
fi

target=./tools/anaconda/anaconda3
if [ -n "$conda" ];then
    if [ -d $target ];then
	echo "anaconda installation already exists in $target"
	exit 1
    fi
    if [ -h $target ];then
	echo "anaconda link already exists in $target"
	exit 1
    fi
    echo "Creating link to anaconda in $target"
    conda=$(readlink -f $conda)
    ln -s $conda $target
fi

target=./tools/kaldi/kaldi
if [ -n "$kaldi" ];then
    if [ -d $target ];then
	echo "anaconda installation already exists in $target"
	exit 1
    fi
    if [ -h $target ];then
	echo "anaconda link already exists in $target"
	exit 1
    fi
    kaldi=$(readlink -f $kaldi)
    echo "Creating link to kaldi in $target"
    ln -s $conda $target
fi
which python
if [ -n "$env" ];then
    cat ./tools/proto_path.sh | \
	sed -e 's@__HYP_ENV__@'$env'@' > ./tools/path.sh

    . ./tools/anaconda/anaconda3/etc/profile.d/conda.sh
    conda activate $env
    x=$(pip freeze | awk 'BEGIN{x=0} /hyperion/ { x=1 } END{ print x }')
    if [ $x -eq 1 ];then
	echo "Hyperion is installed in env $env"
	echo "Recipes will use the installed one"
    else
	echo "Hyperion is not installed in env $env"
	echo "Adding hyperion directory to the PYTHONPATH variable in the recipes"
	echo "export PYTHONPATH=\$HYP_ROOT:\$PYTHONPATH" >> ./tools/path.sh
    fi 
fi
