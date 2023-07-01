import torch
import sys
# arguments example
#

ASR_model = torch.load(sys.argv[1])
LID_model = torch.load(sys.argv[2])
joint_model = torch.load(sys.argv[3])

output_model = sys.argv[4]


def check_update_parameters(joint_state_dict, new_joint_state_dict):
    shape_changed_parameters = []
    unchanged_parameters = []
    changed_parameters = []
    unloaded_parameters = []
    for name, param in joint_state_dict.items():
        new_param = new_joint_state_dict[name].to(param.device)
        if param.shape != new_param.shape:
            shape_changed_parameters.append(name)
        elif torch.all(torch.eq(param, new_param)):
            unchanged_parameters.append(name)
        else:
            changed_parameters.append(name)
    print("Shape changed parameters: {}".format(shape_changed_parameters))
    print("Unchanged parameters: {}".format(unchanged_parameters))
    print("Changed parameters: {}".format(changed_parameters))



def copy_model_parameters(ASR_model, LID_model, joint_model, output_model):
    ASR_state_dict = ASR_model["model_state_dict"]
    LID_state_dict = LID_model["model_state_dict"]

    LID_state_dict = {"module." + name: param for name, param in LID_state_dict.items()} 

    joint_state_dict = joint_model["model_state_dict"]

    hf_feats_update_state_dict = {name: param for name, param in ASR_state_dict.items() if name in joint_state_dict and param.shape == joint_state_dict[name].shape and "hf_feats" in name}
    transducer_update_state_dict = {name: param for name, param in ASR_state_dict.items() if name in joint_state_dict and param.shape == joint_state_dict[name].shape and ("transducer" in name or "film" in name)}
    languageid_update_state_dict = {name: param for name, param in LID_state_dict.items() if name in joint_state_dict and param.shape == joint_state_dict[name].shape and "languageid" in name}
    

    film_update_state_dict = {}
    for name, param in joint_state_dict.items():
        if "linear_scale.weight" in name and "lid_film" in name:
            film_update_state_dict[name] = torch.zeros_like(param)
        elif "linear_scale.bias" in name and "lid_film" in name:
            film_update_state_dict[name] = torch.ones_like(param)
        elif ("linear_shift.weight" in name or "linear_shift.bias" in name) and "lid_film" in name:
            film_update_state_dict[name] = torch.zeros_like(param)
    
    new_joint_state_dict = joint_state_dict.copy()
    new_joint_state_dict.update(hf_feats_update_state_dict)
    new_joint_state_dict.update(transducer_update_state_dict)
    new_joint_state_dict.update(languageid_update_state_dict)
    new_joint_state_dict.update(film_update_state_dict)

    # import pdb;pdb.set_trace()
    
    new_joint_state_dict["module.transducer_fuser"] = ASR_state_dict["module.feat_fuser"]
    new_joint_state_dict["module.languageid_fuser"] = LID_state_dict["module.feat_fuser"]

    
    joint_model["model_state_dict"] = new_joint_state_dict
    joint_model["epoch"] =1

    check_update_parameters(joint_state_dict, new_joint_state_dict)
    torch.save(joint_model, output_model)



copy_model_parameters(ASR_model, LID_model, joint_model, output_model)