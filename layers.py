import torch
import torch.nn as nn


class FrequencyDelayedStack(nn.Module):
    def __init__(self, dims, hook=None):
        super().__init__()
        self.rnn = nn.GRU(dims, dims, batch_first=True)
        self.hidden = nn.Parameter(torch.zeros(1, 1, dims))
        self.hook = hook

    def forward(self, x_time, x_freq, x_cent=None):
        # sum the inputs
        x = x_time + x_freq

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x.size()
        # collapse the first two axes: B & Time
        x = x.view(-1, M, D)

        # Through the RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x, self.hidden.expand(-1, B*T, -1).contiguous())
        if self.hook:
            x.register_hook(self.hook)
        return x.view(B, T, M, D)


class TimeDelayedStack(nn.Module):
    def __init__(self, dims, hook=None):
        super().__init__()
        self.bi_freq_rnn = nn.GRU(dims, dims, bidirectional=True, batch_first=True)
        self.time_rnn = nn.GRU(dims, dims, batch_first=True)
        self.bi_freq_hidden = nn.Parameter(torch.zeros(2, 1, dims))
        self.time_hidden = nn.Parameter(torch.zeros(1, 1, dims))
        self.hook = hook

    def forward(self, x_time):

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x_time.size()

        # Collapse the first two axes
        time_input = x_time.transpose(1, 2).contiguous().view(-1, T, D)  # [B*M, T, D]
        freq_input = x_time.view(-1, M, D)  # [B*T, M, D]

        # Run through the rnns
        self.time_rnn.flatten_parameters()
        x_1, _ = self.time_rnn(time_input,
                               self.time_hidden.expand(-1, B*M, -1).contiguous())
        self.bi_freq_rnn.flatten_parameters()
        x_2_and_3, _ = self.bi_freq_rnn(freq_input,
                                        self.bi_freq_hidden.expand(-1, B*T, -1).contiguous())

        if self.hook:
            x_1.register_hook(self.hook)
            x_2_and_3.register_hook(self.hook)

        # Reshape the first two axes back to original
        x_1 = x_1.view(B, M, T, D).transpose(1, 2)
        x_2_and_3 = x_2_and_3.view(B, T, M, 2 * D)

        # And concatenate for output
        x_time = torch.cat([x_1, x_2_and_3], dim=3)
        return x_time


class CentralizedStack(nn.Module):
    def __init__(self, dims, num_mels):
        self.rnn = nn.GRU(dims * num_mels, dims, batch_first=True)
        self.hidden = nn.Parameter(1, 1, dims)

    def forward(self, x_time):
        B, T, M, D = x_time.size() 
        x_time = x_time.view(B, T, -1)
        out, _ = self.rnn(x_time, self.hidden.expand(-1, B, -1).contiguous())
        return out.unsqueeze(2).expand(B, T, M, D)  # size: B, T, D


class Layer(nn.Module):
    def __init__(self, dims, hook=None):
        super().__init__()
        self.freq_stack = FrequencyDelayedStack(dims, hook)
        self.freq_out = nn.Linear(dims, dims)
        self.time_stack = TimeDelayedStack(dims, hook)
        self.time_out = nn.Linear(3 * dims, dims)

    def forward(self, x):
        # unpack the input tuple
        x_time, x_freq = x

        # grab a residual for x_time
        x_time_res = x_time
        # run through the time delayed stack
        x_time = self.time_stack(x_time)
        # reshape output
        x_time = self.time_out(x_time)
        # connect time residual
        x_time = x_time + x_time_res

        # grab a residual for x_freq
        x_freq_res = x_freq
        # run through the freq delayed stack
        x_freq = self.freq_stack(x_time, x_freq)
        # reshape output TODO: is this even needed?
        x_freq = self.freq_out(x_freq)
        # connect the freq residual
        x_freq = x_freq + x_freq_res
        return x_time, x_freq

