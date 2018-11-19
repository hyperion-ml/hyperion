#!/bin/bash

TARGET=$HOME/usr/local/anaconda3.5
BAZEL=$HOME/usr/local/bazel

# Set env vars
export PATH="$TARGET/bin:$BAZEL/bin:$PATH"
export PYTHONPATH=""

VERS=$1
DEV=$2
WHL_PATH=$PWD/whl/$DEV
WHL=$WHL_PATH/tensorflow-$VERS.0-cp35-cp35m-linux_x86_64.whl

ENV=tensorflow${VERS}_${DEV}

#Install tensorflow inside a conda environment
#create conda enviroment
conda create -n $ENV python=3.5
#activate enviroment
source activate $ENV

conda install anaconda
conda update pip

if [ ! -f $WHL ];then

    export TF_NEED_JEMALLOC=1
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_S3=0
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_COMPUTECPP=0
    export TF_NEED_OPENCL=0
    
    export TF_ENABLE_XLA=1
    export CC_OPT_FLAGS="-march=native"
    #export CC_OPT_FLAGS="-mavx -mavx2 - mfma -mfpmath=both"
    export GCC_HOST_COMPILER_PATH=/usr/bin/gcc


    if [ "$DEV" == "cpu_mkl" ];then
	export TF_NEED_CUDA=0
	export TF_NEED_MKL=1
	export TF_DOWNLOAD_MKL=1
	OPT="--config=mkl --copt=-DEIGEN_USE_VML -c opt --config=opt"
	export LD_LIBRARY_PATH=""
    elif [ "$DEV" == "cpu_nomkl" ];then
	export TF_NEED_CUDA=0
	export TF_NEED_MKL=0
	export TF_DOWNLOAD_MKL=0
	OPT="-c opt --config=opt"
	export LD_LIBRARY_PATH=""
    else
	#I dont know if we need this to find cudnn but just in case
	export CPATH=$HOME/usr/local/cudnn-8.0-v7.0/include:$CPATH
	LD_LIBRARY_PATH=/usr/local/cuda/lib64
	LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$HOME/usr/local/cudnn-8.0-v7.0/lib64:$LD_LIBRARY_PATH
	export TF_NEED_CUDA=1
	export TF_CUDA_VERSION=8.0
	export TF_CUDNN_VERSION=7
	export TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.7,5.2,6.1
	export TF_CUDA_CLANG=0
	export TF_NEED_MKL=0
	export TF_DOWNLOAD_MKL=0
	if [ "$VERS" == "1.7" ];then
	    export CUDA_TOOLKIT_PATH=$HOME/usr/local/cuda
	    fdev=$CUDA_TOOLKIT_PATH/nvvm/libdevice/libdevice.10.bc
	    if [ ! -f $fdev ];then
		fdev0=$CUDA_TOOLKIT_PATH/nvvm/libdevice/libdevice.compute_20.10.bc
		cp $fdev0 $fdev
	    fi
	else
	    export CUDA_TOOLKIT_PATH=/usr/local/cuda
	fi
	export CUDNN_INSTALL_PATH=$HOME/usr/local/cudnn-8.0-v7.0
	
	OPT="--config=opt --config=cuda -c opt"
    fi

    if [ ! -d ./tensorflow ];then
	git clone https://github.com/tensorflow/tensorflow
    fi

    cd tensorflow
    git stash
    git checkout r$VERS

    mkdir -p $WHL_PATH
    bazel clean
    #pip uninstall tensorflow
    ./configure
    echo bazel build $OPT //tensorflow/tools/pip_package:build_pip_package
    bazel build --verbose_failures $OPT //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHL_PATH

    cd -
fi

pip install $WHL

#deactivate enviroment
source deactivate $ENV

