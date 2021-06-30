# Copyright      2019   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

. parse_options.sh || exit 1;

score_dir=$1

#global table

dirs=(plda_2folds plda_snorm_2folds)
cases=("adapt" "adapt s-norm") 

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


