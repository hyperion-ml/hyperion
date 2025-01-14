. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


data=voxceleb2cat_full

xvector_dir=exp/xvectors/fbank80_stmn_ecapatdnn512x3.v3.0.s1/$data
xvector_dir_enroll=exp/xvectors/fbank80_stmn_ecapatdnn512x3.v3.0.s1/voxceleb1_test

exp=$xvector_dir_enroll/cosine_train

if [ $stage -le 1 ];then
  mkdir -p $cluster_dir
  $train_cmd --mem 10G $exp/log/cosine.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-compute-cosine compute_cos \
    --segments-file data/$data/segments.csv \
    --feats-file csv:$xvector_dir/xvector.csv \
    --segments-file-enroll data/voxceleb1_test/segments.csv \
    --feats-file-enroll csv:$xvector_dir_enroll/xvector.csv \
    --output-dir $exp
fi


