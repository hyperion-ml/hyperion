# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

diar_name=diar3b_t-0.9
net_name=3b

. parse_options.sh || exit 1;

be_name=lda300_splday175_v1_train_combined
score_dir=exp/scores/$net_name/${be_name}

#Video table
args="--print_header true"
dirs=(plda plda_gtvad plda_${diar_name})
cases=("w/o diar" "ground-truth diar" "${diar_name}")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    dir_i=$score_dir/${d}_cal_v1
    local/make_table_line.sh $args "${net_name} ${cases[$i]}" $dir_i
    args=""
done

echo ""
