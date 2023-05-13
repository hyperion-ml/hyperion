import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, input_size, condition_size):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the FiLM layer
        super(FiLM, self).__init__()
        self.linear_scale = nn.Linear(condition_size, input_size)
        self.linear_shift = nn.Linear(condition_size, input_size)

    def forward(self, x, condition):
        gamma = self.linear_scale(condition).unsqueeze(2).expand_as(x)
        beta = self.linear_shift(condition).unsqueeze(2).expand_as(x)
        x = x * gamma + beta
        return x



class LSTMWithFiLM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, condition_size, batch_first=True):
        super(LSTMWithFiLM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        self.lstms = nn.ModuleList([nn.LSTM(input_size if i==0 else hidden_size, hidden_size, 1, batch_first=batch_first) for i in range(num_layers)])
        self.films = nn.ModuleList([FiLM(hidden_size, condition_size) for _ in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, states, condition):
        outputs = []
        h, c = states
        new_h, new_c = [], []
        for i, (lstm, film) in enumerate(zip(self.lstms, self.films)):
            x, (h_i, c_i) = lstm(x, (h[i].unsqueeze(0), c[i].unsqueeze(0)))
            x = film(x, condition)
            new_h.append(h_i)
            new_c.append(c_i)
            if i != self.num_layers - 1:
                x = self.dropout_layer(x)
            outputs.append(x)
        new_h = torch.cat(new_h, dim=0)
        new_c = torch.cat(new_c, dim=0)
        return torch.cat(outputs, dim=0), (new_h, new_c)



def initialize_lstm_with_film(lstm_with_film, pretrained_dict):
    # Load pretrained LSTM state_dict
    pretrained_lstm = pretrained_dict['lstm']
    pretrained_num_layers = pretrained_dict['num_layers']

    # Copy weights from pretrained LSTM layers to LSTMWithFiLM
    for i, (lstm, film) in enumerate(zip(lstm_with_film.lstms, lstm_with_film.films)):
        if i < pretrained_num_layers:
            lstm.weight_ih_l0.data.copy_(pretrained_lstm['weight_ih_l' + str(i)])
            lstm.weight_hh_l0.data.copy_(pretrained_lstm['weight_hh_l' + str(i)])
            lstm.bias_ih_l0.data.copy_(pretrained_lstm['bias_ih_l' + str(i)])
            lstm.bias_hh_l0.data.copy_(pretrained_lstm['bias_hh_l' + str(i)])
        else:
            # For extra layers in LSTMWithFiLM, just reset the weights
            nn.init.xavier_uniform_(lstm.weight_ih_l0)
            nn.init.orthogonal_(lstm.weight_hh_l0)
            nn.init.zeros_(lstm.bias_ih_l0)
            nn.init.zeros_(lstm.bias_hh_l0)


# def initialize_lstm_with_film(lstm_with_film, pretrained_lstm):
#     # Copy weights from pretrained LSTM layers to LSTMWithFiLM
#     for i, (lstm, film) in enumerate(zip(lstm_with_film.lstms, lstm_with_film.films)):
#         if i < pretrained_lstm.num_layers:
#             lstm.weight_ih_l0.data.copy_(pretrained_lstm.weight_ih_l[i])
#             lstm.weight_hh_l0.data.copy_(pretrained_lstm.weight_hh_l[i])
#             lstm.bias_ih_l0.data.copy_(pretrained_lstm.bias_ih_l[i])
#             lstm.bias_hh_l0.data.copy_(pretrained_lstm.bias_hh_l[i])
#         else:
#             # For extra layers in LSTMWithFiLM, just reset the weights
#             nn.init.xavier_uniform_(lstm.weight_ih_l0)
#             nn.init.orthogonal_(lstm.weight_hh_l0)
#             nn.init.zeros_(lstm.bias_ih_l0)
#             nn.init.zeros_(lstm.bias_hh_l0)



    # rnn = LSTMWithFiLM(embed_dim, hid_feats, num_layers, rnn_dropout_rate, batch_first=True)