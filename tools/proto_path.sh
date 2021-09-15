# Basic path.sh file to be used in the hyperon recipes

CONDA_ROOT=__CONDA_ROOT__
HYP_ENV=__HYP_ENV__
export LC_ALL=C
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools

# Add Kaldi tools to path
KALDI_ROOT=$TOOLS_ROOT/kaldi
if [ -d "$KALDI_ROOT" ];then
  export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PATH
  [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
  . $KALDI_ROOT/tools/config/common_path.sh
fi

# Add Hyperion tools to path
PATH=$HYP_ROOT/tools/sph2pipe_v2.5:$PATH

if [ -n "$CONDA_ROOT" ];then
  if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    . $CONDA_ROOT/etc/profile.d/conda.sh
  else
    echo "Anaconda not found in $CONDA_ROOT"
    exit 1
  fi
else
  f=$(which python)
  if [ -z "$f" ];then
    echo "Conda installation not found"
    exit 1
  fi
fi

conda activate $HYP_ENV

if [ "$(hostname --domain)" == "cm.gemini" ];then
    module load ffmpeg
fi

export LRU_CACHE_CAPACITY=1 #this will avoid crazy ram memory when using pytorch with cpu, it controls cache of MKLDNN
export HDF5_USE_FILE_LOCKING=FALSE

export MPLBACKEND="agg"
export PATH=$HYP_ROOT/hyperion/bin:$HYP_ROOT/hyp_utils:$PWD/utils:$PATH
export LD_LIBRARY_PATH

wait_file() {
    local file="$1"; shift
    local wait_seconds="${2:-30}"; shift # 10 seconds as default timeout
    for((i=0; i<$wait_seconds; i++)); do
	[ -f $file ] && return 1
	sleep 1s
    done
    return 0
}

export -f wait_file
