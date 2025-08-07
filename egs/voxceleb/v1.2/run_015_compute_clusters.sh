. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


data=voxceleb2cat_full
#data=voxceleb2cat_500_xvector_train
xvector_dir=exp/xvectors/fbank80_stmn_ecapatdnn512x3.v3.0.s1/voxceleb2cat_full

#xvector_dir=exp/xvectors/fbank80_stmn_ecapatdnn512x3.v3.0.s1_voxceleb2cat_500/voxceleb2cat_500_xvector_train
xvector_dir_enroll=exp/xvectors/fbank80_stmn_ecapatdnn512x3.v3.0.s1/voxceleb1_test
nb_clusters=8
option=speaker
exp=$cluster_dir/$data/$option/${nb_clusters}_clusters

if [ $stage -le 1 ];then
  mkdir -p $cluster_dir
  $train_cmd --mem 50G --num-threads 32 $exp/clustering.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-cluster-embeddings $cluster_method --cfg $cluster_cfg \
    --segments-file data/$data/segments.csv \
    --feats-file csv:$xvector_dir/xvector.csv \
    --segments-file-enroll data/voxceleb1_test/segments.csv \
    --feats-file-enroll csv:$xvector_dir_enroll/xvector.csv \
    --output-file $exp/segments_kmeans.csv
fi



