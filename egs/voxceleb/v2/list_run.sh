# Outline: Records for all runs to reproduce the first experimental results
# Rules: SSL-specific scripts' indicies start with 5 (b/c its shape similar to S)



# 1. clone a0b394f8ff18b0cd437a2622ff2b5c6df93d1afa

# 2. run "./make_clsp_links.sh" at the root dir (../../..) to set up dependencies in clsp

# 3. (ING) copy/link required files to run experiments (instead of running scripts before *11):
# * copies:
## cp /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/default_config.sh .
## cp /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/global_conf/dinossl_tuning/config_fbank80_stmn_lresnet34_e256_do0_adam_b128_amp.dinossl_keep_classif_net.v1.sh global_conf/dinossl_tuning/
## cp /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/run_011_train_xvector.dinossl.sh run_511_train_xvector.dinossl.sh
## cp /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/datapath.sh .
## cp /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/cmd.sh . (then, edited)
## cp /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/path.sh .
# * links:
## ln -s /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/data data
## ln -s ../v1/conf conf
## ln -s ../../../hyp_utils hyp_utils
## ln -s hyp_utils/kaldi/utils utils
## ln -s ../v1/local local
## ln -s hyp_utils/kaldi/steps steps
## ln -s ../v1/steps_be steps_be
## ln -s hyp_utils/kaldi/vad steps_fe
## ln -s hyp_utils/feats steps_pyfe
## ln -s hyp_utils/xvectors steps_xvec
# * copies of fixed codes after the above:
## cp ${dir_dino}/steps_be/eval_be_v1.sh steps_be/eval_be_v1.sh
## cp ${dir_dino}/steps_xvec/extract_xvectors_from_wav.sh steps_xvec/extract_xvectors_from_wav.sh



# 4. (ING) run experiments
## front-end
bash run_511_train_xvector.dinossl.sh --config-file global_conf/dinossl_tuning/config_fbank80_stmn_lresnet34_e256_do0_adam_b128_amp.dinossl_keep_classif_net.v1.sh 2>&1 | tee exp/log/run_011_train_xvector.dinossl.config_fbank80_stmn_lresnet34_e256_do0_adam_b128_amp.dinossl_keep_classif_net.v1.log
## back-end
(ING) bash run.backend_v1.sh --config-file global_conf/dinossl_tuning/config_fbank80_stmn_lresnet34_e256_do0_adam_b128_amp.dinossl_keep_classif_net.v1.sh 2>&1 | tee exp/log/run.backend_v1.model_ep0070.dinossl_keep_classif_net.v1.log
