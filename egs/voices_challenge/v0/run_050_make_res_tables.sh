# Copyright      2019   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_name=3b

. parse_options.sh || exit 1;

be_name=lda300_splday175_v1_train_combined
score_dir=exp/scores/$net_name/${be_name}

#global table

dirs=(plda plda_snorm plda_2folds plda_snorm_2folds)
cases=("cal-1fold" "s-norm cal-1fold" "" "s-norm") 

echo "All"
args="--print_header true"
nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    dir_i=$score_dir/${d}_cal_v1
    local/make_table_line.sh $args "${net_name} ${cases[$i]}" $dir_i
    args=""
done

echo ""

#conditions table
for cond_type in noise mic_type mic_pos orientation
do
    echo $cond_type
    args="--print_header true"
    for((i=0;i<$nc;i++))
    do
	d=${dirs[$i]}
	dir_i=$score_dir/${d}_cal_v1
	local/make_table_line.sh --cond_type $cond_type $args "${net_name} ${cases[$i]}" $dir_i
	args=""
    done
    echo ""
done


