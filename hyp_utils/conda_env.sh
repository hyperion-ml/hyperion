#!/bin/bash
# Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ -f "path.sh" ];then
    # SGE runs ~/.bashrc so it may change the order of env vars
    # We run path.sh to be sure we have the right PATH
    . path.sh
fi
set -e
num_gpus=0
if [ -n "$HYP_ENV" ];then
    conda_env=$HYP_ENV
else
    conda_env=base
fi

while true
do
    if [ "$1" == "--num-gpus" ];then
	shift;
	num_gpus=$1
	shift;
    elif [ "$1" == "--conda-env" ];then
	shift;
	conda_env=$1
	shift;
    else
	break
    fi
done

if [ $# -lt 1 ];then
    echo "Usage: conda_env.sh [--num-gpus n>=0] [--conda-env <conda-env>] python_program.py [args1] [arg2] ..."
    echo "Wrapper over python to "
    echo " - activate a conda environment "
    echo " - set CUDA_VISIBLE_DEVICES automatically"
    echo "Ex using kaldi's queue.pl utility:"
    echo ""
    echo "export HYP_ENV=pytorch1.0_cuda9.0"
    echo "queue.pl --gpu 1 -V log_file conda_env.sh --num-gpus 1 --conda-env $HYP_ENV train-dnn.py --lr 0.1"
    exit 0
fi
# echo "PATH=$PATH"
# echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
# export LRU_CACHE_CAPACITY=1
# echo "LRU_CACHE_CAPACITY=$LRU_CACHE_CAPACITY"

conda activate $conda_env
command="python"
if [ $num_gpus -gt 0 ];then
    # set CUDA_VISIBLE_DEVICES
  if [ ! -z "$SGE_HGR_gpu" ]; then
    echo "SGE_HGR_gpu=$SGE_HGR_gpu"
    export CUDA_VISIBLE_DEVICES=$(echo $SGE_HGR_gpu | sed 's@ @,@g')
  else
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
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  export TORCH_DISTRIBUTED_DEBUG=DETAIL #variable to find unused parameters
  if [ $num_gpus -gt 1 ];then
    # export CUDA_LAUNCH_BLOCKING=1
    [[ $(type -P "$torchrun") ]] && command="torchrun" \
	|| command="python -m torch.distributed.run"
    command="$command --nproc_per_node=$num_gpus --standalone --nnodes=1"
  fi
fi

py_exec=$(which $1)
shift

$command $py_exec "$@"

conda deactivate 

