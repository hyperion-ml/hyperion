. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file



xvector_dir=exp/xvectors/fbank80_stmn_ecapatdnn512x3.v3.0.s1_voxceleb2cat_500/voxceleb2cat_500_xvector_train

if [ $stage -le 1 ];then
  echo "Cluster "
  mkdir -p $cluster_dir
  $train_cmd --mem 50G --num-threads 32 $cluster_dir/clustering.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-cluster-embeddings $cluster_method --cfg $cluster_cfg \
    --segments-file data/voxceleb2cat_500_xvector_train/segments.csv \
    --feats-file csv:$xvector_dir/xvector.csv \
    --output-file $cluster_dir/voxceleb2cat_500_xvector_train/segments.csv
fi


