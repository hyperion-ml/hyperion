#!/bin/bash

# This script is to run backend exp. for dinossl: xvector extraction, (backend training), and SV eval

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
dinossl_xvec_loc="f" # position we extract xvectors: choices=["f", "dinohead_mlp","dinohead_l2norm","dinohead_linear"]

. parse_options.sh || exit 1;
. $config_file

# This is optional. If you want to extract x-vector embedding from one of the
# layers in DINOHead, this is required.
if [[ ${dinossl_xvec_loc} != "f" ]]; then
    dinossl_cfg="${nnet_dir}/config_dinosslonly_${dinossl_xvec_loc}.yaml"
    echo "Optional Stage: Creating ${dinossl_cfg}"
    if [ ! -s ${dinossl_cfg} ]; then
        grep "^dinossl" ${nnet_dir}/config.yaml > ${dinossl_cfg}
        echo "dinossl_xvec_loc: ${dinossl_xvec_loc}" >> ${dinossl_cfg}
    fi
fi

if [ $stage -le 1 ]; then
    # add a model_cfg (in the first version of dinossl code, I did not save
    # them in training): TODO: THIS STAGE WILL BE REMOVED IN PR AND CHANGE
    # STAGE INDICES
    if [[ ${nnet} == *manual.pth ]]; then
        echo "Stage 1: Adding model_cfg ..."
        bash run_add_model_cfg.sh --ngpu 1 --config-file ${config_file}
    fi
fi

if [ $stage -le 2 ]; then
    # extract xvectors only for voxceleb1_test: this is the combination of
    # voxceleb1 dev and test subsets in the official naming
    echo "Stage 2: Extracting xvectors ..."
    if [[ -n ${dinossl_cfg} ]]; then
        echo "Extracting xvectors with ${dinossl_cfg}"
        bash run_030_extract_xvectors.sh --stage 2 --config-file ${config_file} --dinossl_cfg ${dinossl_cfg}
    else
        echo "Extracting xvectors right before DINOHead"
        bash run_030_extract_xvectors.sh --stage 2 --config-file ${config_file}
    fi
fi

if [ $stage -le 3 ]; then
    # run back-end eval
    echo "Stage 3: Backend (train) and evalution ..."
    bash run_040_eval_be.vox1only.sh --config-file ${config_file} --dinossl_xvec_loc ${dinossl_xvec_loc}
fi
