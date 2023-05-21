import torch
import sys
# arguments example
# ASR_model = 'exp/transducer_nnets/wav2vec2xlsr300m_rnnt_k2_pruned.v4.0_13_langs_weighted_8000_bpe.s2/model_ep0010.pth'
# LID_model = "exp/transducer_nnets/wav2vec2xlsr300m_rnnt_k2_pruned_film.v1.0_13_langs_weighted_8000_bpe.s1_initial/model_ep0000.pth"
# output_model = "model_initialized.pth"

# python local/initailize_lid_model.py /gspvolume/home/ec2-user/hyperion/egs/commonvoice/v1/exp/transducer_nnets/wav2vec2xlsr300m_rnnt_k2_pruned.v4.0_13_langs_weighted_8000_bpe.s2/model_ep0010.pth /gspvolume/home/ec2-user/hyperion/egs/commonvoice/v1/exp/resnet1d_nnets/wav2vec2xlsr300m_resnet1d_v4.2_13_langs.s1/model_ep0003.pth  /gspvolume/home/ec2-user/hyperion/egs/commonvoice/v1/exp/resnet1d_nnets/wav2vec2xlsr300m_resnet1d_v4.2_13_langs.s3/model_ep0001.pth 

ASR_model = torch.load(sys.argv[1])
LID_model = torch.load(sys.argv[2])

output_model = sys.argv[3]


def copy_model_parameters(ASR_model, LID_model):
    ASR_state_dict = ASR_model["model_state_dict"]
    LID_state_dict = LID_model["model_state_dict"]

    update_state_dict = {name: param for name, param in ASR_state_dict.items() if name in LID_state_dict and param.shape == LID_state_dict[name].shape and "hf_feats" in name}
    # remove feature fuser
    
    new_LID_state_dict = LID_state_dict.copy()
    new_LID_state_dict.update(update_state_dict)
    
    LID_model["model_state_dict"] = new_LID_state_dict

    unchanged_parameters = []
    changed_parameters = []
    unloaded_parameters = []
    for name, param in LID_state_dict.items():
        if torch.all(torch.eq(param, new_LID_state_dict[name])):
            unchanged_parameters.append(name)
        else:
            changed_parameters.append(name)

    for name, param in ASR_state_dict.items():
        if name not in changed_parameters:
            unloaded_parameters.append(name)

    print(f"Unchanged parameters: {unchanged_parameters}")
    print(f"Unloaded parameters: {unloaded_parameters}")
    print(f"Changed parameters: {changed_parameters}")
    LID_model["epoch"] =1
    torch.save(LID_model, output_model)



copy_model_parameters(ASR_model, LID_model)