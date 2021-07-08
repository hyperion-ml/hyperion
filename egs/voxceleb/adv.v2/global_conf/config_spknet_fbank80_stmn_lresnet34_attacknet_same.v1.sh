# Experiments using LResNet34 for x-vector extractor and for attack signature extractor
# We use only sucessful attacks
# We use attack SNR in (-100, 100) for train and test

# Victim model speaker LResNet34 x-vector configuration
spknet_command=resnet
spknet_data=voxceleb2cat_train
spknet_config=conf/lresnet34_spknet.yaml
spknet_batch_size_1gpu=128
spknet_eff_batch_size=512 # effective batch size
spknet_name=lresnet34
spknet_dir=exp/xvector_nnets/$spknet_name
spknet=$spknet_dir/model_ep0070.pth

# SpkID Attacks configuration
feat_config=conf/fbank80_stmn_16k.yaml
p_attack=0.25 #will try attacks in 25% of utterances
attacks_common_opts="--save-failed --save-benign" #save failed attacks also

# SpkVerif Attacks configuration
p_tar_attack=0.1
p_non_attack=0.1
spkv_attacks_common_opts="--save-failed" #save failed attacks also


# Attack model LResNet34 configuration
sign_nnet_command=resnet
sign_nnet_config=conf/lresnet34_atnet.yaml
sign_nnet_batch_size_1gpu=128
sign_nnet_eff_batch_size=512 # effective batch size
sign_nnet_name=lresnet34

# SNRs in -100, 100
train_max_snr=100
train_min_snr=-100
test_max_snr=100
test_min_snr=-100
# We only uses succesful attacks (attacks that changed the label)
train_cat=success
test_cat=success

# Splits options
# train and test on succesful attacks only, all SNR values
attack_type_split_opts="--train-success-category $train_cat --test-success-category $test_cat \
--train-max-snr $train_max_snr --train-min-snr $train_min_snr --test-max-snr $test_max_snr --test-min-snr $test_min_snr"
threat_model_split_opts="--train-success-category $train_cat --test-success-category $test_cat \
--train-max-snr $train_max_snr --train-min-snr $train_min_snr --test-max-snr $test_max_snr --test-min-snr $test_min_snr"
# for SNR we use same train/test SNR
snr_split_opts="--train-success-category $train_cat --test-success-category $test_cat \
--train-max-snr $test_max_snr --train-min-snr $test_min_snr --test-max-snr $test_max_snr --test-min-snr $test_min_snr"


# Experiment labels for experiments of attack classification with all attacks known
attack_type_split_tag="exp_attack_type_allknown"
snr_split_tag="exp_attack_snr_allknown"
threat_model_split_tag="exp_attack_threat_model_allknown"


# Known/Unknown attacks splits
known_attacks="fgsm iter-fgsm pgd-linf pgd-l1 pgd-l2"
unknown_attacks="cw-l2 cw-linf cw-l0"

# Experiment labels for datasets to train signatures with a subset of known attacks
sk_attack_type_split_tag="exp_attack_type_someknown"
sk_snr_split_tag="exp_attack_snr_someknown"
sk_threat_model_split_tag="exp_attack_threat_model_someknown"

# Experiment labels for attack verification with same attacks known and some unknown
attack_type_verif_split_tag="exp_attack_type_verif"
snr_verif_split_tag="exp_attack_snr_verif"
threat_model_verif_split_tag="exp_attack_threat_model_verif"

# Select attacks for attack verification, options are shared for the 3 tasks
# We use only successful attacks with all SNRs
verif_split_opts="--success-category $test_cat --max-snr $test_max_snr --min-snr $test_min_snr"

# Select attacks for attack novelty detection
# We use only successful attacks with all SNRs
novelty_split_opts="--success-category $test_cat --max-snr $test_max_snr --min-snr $test_min_snr"
novelty_split_tag="exp_attack_type_novelty"

# Experiment labels for experiments on attacks against speaker verification task
# Here we just do attack classification assuming all attacks known
spkverif_attack_type_split_tag="exp_spkverif_attack_type_allknown"
spkverif_snr_split_tag="exp_spkverif_attack_snr_allknown"
spkverif_threat_model_split_tag="exp_spkverif_attack_threat_model_allknown"
spkverif_split_opts="--test-success-category $test_cat --test-max-snr $test_max_snr --test-min-snr $test_min_snr"
