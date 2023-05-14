import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, input_size, condition_size):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the FiLM layer
        super(FiLM, self).__init__()
        self.linear_scale = nn.Linear(condition_size, input_size)
        self.linear_shift = nn.Linear(condition_size, input_size)

    def forward(self, x, lang_condition):
        if x.ndim == 3:
            gamma = self.linear_scale(lang_condition).unsqueeze(1).expand_as(x)
            beta = self.linear_shift(lang_condition).unsqueeze(1).expand_as(x)
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = self.linear_scale(lang_condition).unsqueeze(1).unsqueeze(2).expand_as(x)
            beta = self.linear_shift(lang_condition).unsqueeze(1).unsqueeze(2).expand_as(x)
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

    def forward(self, x, states, lang_condition):
        outputs = []
        new_h, new_c = [], []
        for i, (lstm, film) in enumerate(zip(self.lstms, self.films)):
            if states:
                x, (h_i, c_i) = lstm(x, (states[0][i].unsqueeze(0), states[1][i].unsqueeze(0)))
            else:
                x, (h_i, c_i) = lstm(x)
            x = film(x, lang_condition)
            new_h.append(h_i)
            new_c.append(c_i)
            if i != self.num_layers - 1:
                x = self.dropout_layer(x)
            outputs.append(x)
        new_h = torch.cat(new_h, dim=0)
        new_c = torch.cat(new_c, dim=0)
        return x, (new_h, new_c)

