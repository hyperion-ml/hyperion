import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, input_size, condition_size, film_type="linear"):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the FiLM layer
        super(FiLM, self).__init__()
        if film_type == "tanh":
            self.linear_scale = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_shift = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
        elif film_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)

    def forward(self, x, lang_condition):
        # import pdb; pdb.set_trace()
        if x.ndim == 3:
            gamma = self.linear_scale(lang_condition).unsqueeze(1).expand_as(x)
            beta = self.linear_shift(lang_condition).unsqueeze(1).expand_as(x)
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = self.linear_scale(lang_condition).unsqueeze(1).unsqueeze(2).expand_as(x)
            beta = self.linear_shift(lang_condition).unsqueeze(1).unsqueeze(2).expand_as(x)
            x = x * gamma + beta
        return x



class RNNWithFiLM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, condition_size, batch_first=True, rnn_type="lstm", film_type="tanh", film_cond_type="one-hot"):
        super(RNNWithFiLM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first 
        self.rnn_type = rnn_type
        self.film_type = film_type
        self.film_cond_type = film_cond_type

        if self.rnn_type == "lstm":
            self.lstms = nn.ModuleList([nn.LSTM(input_size if i==0 else hidden_size, hidden_size, 1, batch_first=batch_first) for i in range(num_layers)])
        elif self.rnn_type == "gru":
            self.grus = nn.ModuleList([nn.GRU(input_size if i==0 else hidden_size, hidden_size, 1, batch_first=batch_first) for i in range(num_layers)])

        if self.film_cond_type == "one-hot":
            self.films = nn.ModuleList([FiLM(hidden_size, condition_size, film_type) for _ in range(num_layers)])
        else:
            self.films = nn.ModuleList([FiLM(hidden_size, condition_size, film_type) for _ in range(num_layers)])
            self.lid_films = nn.ModuleList([FiLM(hidden_size, condition_size, film_type) for _ in range(num_layers)])

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, states, lang_condition):
        outputs = []
        new_h, new_c = [], []
        if self.rnn_type == "lstm":
            rnns = self.lstms
        elif self.rnn_type == "gru":
            rnns = self.grus
            
        if self.film_cond_type == "one-hot":
            films = self.films
        else:
            films = self.lid_films

        for i, (rnn, film) in enumerate(zip(rnns, films)):
            if states:
                x, (h_i, c_i) = rnn(x, (states[0][i].unsqueeze(0), states[1][i].unsqueeze(0)))
            else:
                x, (h_i, c_i) = rnn(x)
            x = film(x, lang_condition)
            new_h.append(h_i)
            new_c.append(c_i)
            if i != self.num_layers - 1:
                x = self.dropout_layer(x)
            outputs.append(x)
        new_h = torch.cat(new_h, dim=0)
        new_c = torch.cat(new_c, dim=0)
        return x, (new_h, new_c)


class RNNWithFiLMResidual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, condition_size, batch_first=True, rnn_type="lstm_residual", film_type="linear"):
        super(RNNWithFiLMResidual, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first 
        self.rnn_type = rnn_type
        if self.rnn_type == "lstm_residual":
            self.lstms = nn.ModuleList([nn.LSTM(input_size if i==0 else hidden_size, hidden_size, 1, batch_first=batch_first) for i in range(num_layers)])
        elif self.rnn_type == "gru_residual":
            self.grus = nn.ModuleList([nn.GRU(input_size if i==0 else hidden_size, hidden_size, 1, batch_first=batch_first) for i in range(num_layers)])
        self.film_type = film_type
        self.films = nn.ModuleList([FiLM(hidden_size, condition_size, film_type)  for _ in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, states, lang_condition):
        outputs = []
        new_h, new_c = [], []

        if self.rnn_type == "lstm_residual":
            rnns = self.lstms
        elif self.rnn_type == "gru_residual":
            rnns = self.grus
            
        for i, (rnn, film) in enumerate(zip(rnns, self.films)):
            if states:
                x, (h_i, c_i) = rnn(x, (states[0][i].unsqueeze(0), states[1][i].unsqueeze(0)))
            else:
                x, (h_i, c_i) = rnn(x)
            x = film(x, lang_condition)
            if i != 0:
                x = x + residual
            residual = x
            new_h.append(h_i)
            new_c.append(c_i)
            if i != self.num_layers - 1:
                x = self.dropout_layer(x)
            outputs.append(x)
        new_h = torch.cat(new_h, dim=0)
        new_c = torch.cat(new_c, dim=0)
        return x, (new_h, new_c)

