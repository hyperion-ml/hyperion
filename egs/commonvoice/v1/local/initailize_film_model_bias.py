import torch
import sys

# arguments example
# pretrained_model = 'exp/transducer_nnets/wav2vec2xlsr300m_rnnt_k2_pruned.v4.0_13_langs_weighted_8000_bpe.s2/model_ep0010.pth'
# film_model = "exp/transducer_nnets/wav2vec2xlsr300m_rnnt_k2_pruned_film.v1.0_13_langs_weighted_8000_bpe.s1_initial/model_ep0000.pth"
# output_model = "model_initialized.pth"

pretrained_model = torch.load(sys.argv[1])
film_model = torch.load(sys.argv[2])

output_model = sys.argv[3]


def update_film_lstm_parameters(film_state_dict, pretrained_state_dict):
    for i in range(2):
        film_state_dict["module.transducer.decoder.predictor.rnn.lstms." + str(i) + ".weight_ih_l0"] = pretrained_state_dict['module.transducer.decoder.predictor.rnn.weight_ih_l' + str(i)].clone()
        film_state_dict["module.transducer.decoder.predictor.rnn.lstms." + str(i) + ".weight_hh_l0"]= pretrained_state_dict['module.transducer.decoder.predictor.rnn.weight_hh_l' + str(i)].clone()
        film_state_dict["module.transducer.decoder.predictor.rnn.lstms." + str(i) + ".bias_ih_l0"]= pretrained_state_dict['module.transducer.decoder.predictor.rnn.bias_ih_l' + str(i)].clone()
        film_state_dict["module.transducer.decoder.predictor.rnn.lstms." + str(i) + ".bias_hh_l0"]= pretrained_state_dict['module.transducer.decoder.predictor.rnn.bias_hh_l' + str(i)].clone()
    return film_state_dict

def copy_model_parameters(pretrained_model, film_model):
    pretrained_state_dict = pretrained_model["model_state_dict"]
    film_state_dict = film_model["model_state_dict"]
    update_state_dict = {name: param for name, param in pretrained_state_dict.items() if name in film_state_dict and param.shape == film_state_dict[name].shape}

    film_update_state_dict = {}
    for name, param in film_state_dict.items():
        if "linear_scale.weight" in name:
            film_update_state_dict[name] = torch.zeros_like(param)
        elif "linear_scale.bias" in name:
            film_update_state_dict[name] = torch.ones_like(param)
        elif "linear_shift.weight" in name or "linear_shift.bias" in name:
            film_update_state_dict[name] = torch.zeros_like(param)
    # import pdb; pdb.set_trace()
    new_film_state_dict = film_state_dict.copy()
    new_film_state_dict.update(update_state_dict)
    new_film_state_dict.update(film_update_state_dict)


    new_film_state_dict = update_film_lstm_parameters(new_film_state_dict, pretrained_state_dict)

    film_model["model_state_dict"] = new_film_state_dict

    unchanged_parameters = []
    changed_parameters = []
    unloaded_parameters = []
    for name, param in film_state_dict.items():
        if torch.all(torch.eq(param, new_film_state_dict[name])):
            unchanged_parameters.append(name)
        else:
            changed_parameters.append(name)

    for name, param in pretrained_state_dict.items():
        if name not in changed_parameters:
            unloaded_parameters.append(name)

    print(f"Unchanged parameters: {unchanged_parameters}")
    print(f"Unloaded parameters: {unloaded_parameters}")
    print(f"Changed parameters: {changed_parameters}")
    film_model["epoch"] =1
    torch.save(film_model, output_model)



unchanged_parameters = copy_model_parameters(pretrained_model, film_model)