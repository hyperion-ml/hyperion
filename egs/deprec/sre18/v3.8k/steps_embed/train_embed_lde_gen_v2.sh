#!/bin/bash

cmd=run.pl

vers=1a

x_dim=80
y_dim=600

h_y_dim=512
h_t_dim=256
num_layers_y=5
num_layers_t=3

min_frames=300
max_frames=500
overlap=1000
gen_method=random

num_comp=64
lr=0.0005
init_f='normal'
init_s=0.5
init_mode=fan_in
act=relu
ipe=10
p=0.25
preproc_file=""
batch_size=96
epochs=100
lr_patience=2
lr_monitor=val_loss
patience=10
pooling="lde"
lr_decay=0
balanced="false"
lr_factor=0.5
min_lr=1e-6
nepc=1
nepu=1

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

feats_file=$1
listdir=$2
modeldir=$3

if [ -f $modeldir/model.json ];then
    exit 0
fi

mkdir -p $modeldir

embed_net1=$modeldir/net1.json
embed_net2=$modeldir/net2.json
context_file=$modeldir/context

train_list=$listdir/train.scp
val_list=$listdir/val.scp
class_list=$listdir/class2int

t_dim=$(wc -l $class_list | awk '{ print $1}')

train_args=""
net_args=""

if [ -n "$preproc_file" ]; then
    train_args="--preproc-file $preproc_file"
fi

if [ "$balanced" == "true" ]; then
    train_args="$train_args --class-weight balanced"
fi


if [ ! -f $modeldir/model.0000.json ];then 

    if [ "$KERAS_BACKEND" == "tensorflow" ];then
	source activate $TFCPU
    else
	export THEANO_FLAGS="floatX=float32,device=cpu"
    fi

    echo "$0 [info] creatings nets for $modeldir"

    run.pl $modeldir/create_nets.log \
	   steps_embed/nets/create_nets_embed_lde_v${vers}.py --x-dim $x_dim --y-dim $y_dim --t-dim $t_dim \
	   --h-y-dim $h_y_dim \
	   --h-t-dim $h_t_dim \
	   --num-layers-y $num_layers_y \
	   --num-layers-t $num_layers_t \
	   --num-comp $num_comp \
	   --out-path $modeldir $net_args \
	   --dropout $p --act $act \
	   --init-f $init_f --init-mode $init_mode --init-s $init_s
    
    if [ "$KERAS_BACKEND" == "tensorflow" ];then
	source deactivate $TFCPU
    fi

fi


context=$(cat $context_file)

if [ "$KERAS_BACKEND" == "tensorflow" ];then
    source activate $TFGPU
else
    export THEANO_FLAGS="floatX=float32,device=cuda,optimizer=fast_run,dnn.enabled=True,allow_gc=True,warn_float64=warn"
fi

MY_PYTHON=$(which python)
PROGRAM=$(which keras-train-cat-seq-embed-lde-gen-v2.py)

echo "$0 [info] start training embeddings for $modeldir"
$cmd $modeldir/train_embed.log \
     $MY_PYTHON $PROGRAM \
     --data-path scp:$feats_file \
     --train-list $train_list \
     --val-list $val_list \
     --class-list $class_list \
     --prepool-net $embed_net1 \
     --postpool-net $embed_net2 \
     --pooling $pooling \
     --lde-num-comp $num_comp \
     --output-path $modeldir \
     --left-context $context \
     --right-context $context \
     --begin-context 0 --end-context 0 \
     --max-seq-length $max_frames \
     --min-seq-length $min_frames \
     --num-egs-per-class $nepc \
     --num-egs-per-utt $nepu \
     --batch-size $batch_size \
     --lr $lr \
     --iters-per-epoch $ipe \
     --epochs $epochs \
     --patience $patience \
     --lr-decay $lr_decay \
     --lr-patience $lr_patience \
     --lr-factor $lr_factor \
     --lr-monitor $lr_monitor \
     --min-lr $min_lr \
     --save-all-epochs --log-append\
     $train_args

if [ "$KERAS_BACKEND" == "tensorflow" ];then
    source deactivate $TFGPU
fi

wait_file $modeldir/model.json &&  echo "$0 [error] $modeldir/model.json not created" && exit 1

echo "$0 [info] end training embeddings for $modeldir"
