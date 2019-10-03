import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, cell_type, hook=None):
        self.rnn_cell = cell_type(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.hook = hook
        if cell_type == nn.LSTMCell:
            self.lstm = True
        else:
            self.lstm = False

    def forward(self, x):
        batch_size, seq_len, input_size = x.size(0)
        h = x.new_zeros(batch_size, self.hidden_size)
        if self.lstm:
            c = x.new_zeros(batch_size, self.hidden_size)

        for i in range(seq_len):
            if self.hook:
                h.register_hook(self.hook)
            if self.lstm:
                h, c = self.rnn_cell(x[:, i, :], (h, c))
            else:
                h = self.rnn_cell(x[:, i, :], h)

        return h
