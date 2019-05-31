#!/bin/bash
# Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ -f "path.sh" ];then
    # SGE runs ~/.bashrc so it may change the order of env vars
    # We run path.sh to be sure we have the right PATH
    . path.sh
fi

num_gpus=0

if [ "$1" == "--num-gpus" ];then
    shift;
    num_gpus=$1
    shift;
fi

if [ $# -lt 1 ];then
    echo "Usage: torch.sh [--num-gpus n>=0] python_program.py [args1] [arg2] ..."
    echo "Wrapper over python to "
    echo " - activate a conda environment with pytorch installed "
    echo " - set CUDA_VISIBLE_DEVICES automatically"
    echo "The variable TORCH must contain the name of the conda environment"
    echo "Ex using kaldi's queue.pl utility:"
    echo ""
    echo "export TORCH=pytorch1.0_cuda9.0"
    echo "queue.pl --gpu 1 -V log_file torch.sh --num-gpus 1 train-dnn.py --lr 0.1"
    exit 0
fi
echo $PATH

# get conda version
#CONDA_VERS=$(conda -V | awk '{ split($2,f,"."); v=f[1]f[2]; print v}')
CONDA_VERS44=false
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ];then
    # conda version >= 4.4
    CONDA_VERS44=true
    # this is needed so that "conda activate" works
    . $(conda info --base)/etc/profile.d/conda.sh
fi

#we activate conda env if TORCH var is not empty
if [ ! -z "$TORCH" ];then
    
    [ "$CONDA_VERS44" = true ] && conda activate $TORCH || source activate $TORCH
    # if[ "$CONDA_VERS44" = true ];then
    # 	# for conda version >= 4.4
    # 	conda activate $TORCH
    # else
    # 	# for conda version < 4.4
    # 	source activate $TORCH

    # fi
fi

if [ $num_gpus -gt 0 ];then
    # seach location of free-gpu program in the PATH or hyp_utils directory
    free_gpu=$(which free-gpu)
    if [ -z "$free_gpu" ];then
	free_gpu=$(which hyp_utils/free-gpu)
    fi
    
    if [ ! -z "$free_gpu" ];then
	# if free-gpu found set env var, otherwise we assume that you can use any gpu
	export CUDA_VISIBLE_DEVICES=$($free_gpu -n $num_gpus)
    fi
fi

echo $CUDA_VISIBLE_DEVICES
python "$@"

if [ ! -z "$TORCH" ];then
    [ "$CONDA_VERS44" = true ] && conda deactivate || source deactivate
    # if [ "$CONDA_VERS44" == true ];then
    # 	conda deactivate

    # else
    # 	source deactivate
    # fi
fi
