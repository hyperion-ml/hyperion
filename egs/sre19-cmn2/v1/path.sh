
export SRE19_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$SRE19_ROOT/tools

export KALDI_ROOT=$TOOLS_ROOT/kaldi/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

HYP_ROOT=$TOOLS_ROOT/hyperion/hyperion

#Anaconda env
CONDA_ROOT=$TOOLS_ROOT/anaconda/anaconda3
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    set +u
    #for conda version >=4.4 do    
    . $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate
else
    #for conda version <4.4 do 
    PATH=$CONDA_ROOT/bin:$PATH
fi

if [ "$(hostname --domain)" == "cm.gemini" ];then
    module load cuda10.0/toolkit
    module load ffmpeg
else
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
    if [ ! -d /usr/local/cuda/lib64 ]; then
	LD_LIBRARY_PATH=$HOME/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    fi
fi

export MPLBACKEND="agg"
export PATH=$HYP_ROOT/hyperion/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HYP_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH
export LC_ALL=C

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
