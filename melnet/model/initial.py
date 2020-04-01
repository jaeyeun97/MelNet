import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyDelayedStack(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rnn = nn.GRU(width, width, batch_first=True)

    def forward(self, x):
        B, T, M, D = x.size()
        x = x.view(-1, M, D)
        x, _ = self.rnn(x)
        return x.view(B, T, M, D)


class TimeDelayedStack(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.bi_freq_rnn = nn.GRU(width, width, bidirectional=True, batch_first=True)
        self.time_rnn = nn.GRU(width, width, batch_first=True)
        self.hidden_states = dict()

    def forward(self, x_time, entries, flag_lasts):

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x_time.size()

        # Collapse the first two axes
        time_input = x_time.transpose(1, 2).contiguous().view(-1, T, D)  # [B*M, T, D]
        freq_input = x_time.view(-1, M, D)  # [B*T, M, D]

        # Freq RNN
        x_2_and_3, _ = self.bi_freq_rnn(freq_input)

        # Time RNN 
        hidden_state = torch.stack([self.hidden_states.setdefault(entries[i], time_input.new_zeros(M, D))
                                    for i in range(B)], dim=0) # (B, M, D)
        hidden_state = hidden_state.view(B * M, D).unsqueeze(0) # (1, B * M, D)
        x_1, hidden_state = self.time_rnn(time_input)

        # Reshape the first two axes back to original
        x_1 = x_1.view(B, M, T, D).transpose(1, 2)
        hidden_state = hidden_state.squeeze(0).view(B, M, D)
        x_2_and_3 = x_2_and_3.view(B, T, M, 2 * D)

        # And concatenate for output
        x_time = torch.cat([x_1, x_2_and_3], dim=3)

        for i in range(B):
            if flag_lasts[i]:
                del self.hidden_states[entries[i]]
            else:
                self.hidden_states[entries[i]] = hidden_state[i, :, :].clone().detach()

        return x_time # B, T, M, 3D


class CentralizedStack(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.hidden_states = dict()

    def forward(self, x, entries, flag_lasts):
        B, T, D = x.size()
        h = torch.stack([self.hidden_states.setdefault(entries[i], x.new_zeros(D))
                         for i in range(B)], dim=0).unsqueeze(0) # (1, B, D)
        x, h = self.rnn(x, h)

        for i in range(B):
            if flag_lasts[i]:
                del self.hidden_states[entries[i]]
            else:
                self.hidden_states[entries[i]] = h[0, i, :].clone().detach()

        return x # size: B, T, D


class Layer(nn.Module):
    def __init__(self, width, central=True):
        super().__init__()
        self.freq_stack = FrequencyDelayedStack(width)
        self.freq_out = nn.Linear(width, width)
        if central:
            self.central_stack = CentralizedStack(width)
            self.central_out = nn.Linear(width, width)
        self.central = central
        self.time_stack = TimeDelayedStack(width)
        self.time_out = nn.Linear(3 * width, width)

    def forward(self, x_time, x_central, x_freq, entries, flag_lasts):
        x_time = x_time + self.time_out(self.time_stack(x_time, entries, flag_lasts))
        if self.central:
            x_central = x_central + self.central_out(self.central_stack(x_central, entries, flag_lasts))
        # run through the freq delayed stack
        x_freq_stack = x_time + x_freq
        if self.central:
            x_freq_stack = x_freq_stack + x_central.unsqueeze(2).expand_as(x_freq)
        x_freq_stack = self.freq_stack(x_freq_stack)
        x_freq = x_freq + self.freq_out(x_freq_stack)
        return x_time, x_central, x_freq


class InitialTier(nn.Module):
    def __init__(self, width, n_freq, n_layers, n_mixtures, central=True, cond_size=0):
        super().__init__()
        # Input layers
        self.freq_input = nn.Linear(1, width)
        self.time_input = nn.Linear(1, width)
        if central:
            self.central_input = nn.Linear(n_freq, width)
        self.central = central

        if cond_size != 0:
            # Paper states that there are two condition networks: W^t_z, W^f_z
            self.cond_freq = nn.Linear(cond_size, width)
            self.cond_time = nn.Linear(cond_size, width)
            self.c_freq = None
            self.c_time = None
            self.cond = True
        else:
            self.cond = False

        # Main layers
        self.layers = nn.ModuleList([Layer(width,  central) for i in range(n_layers)])

        # Output layer
        self.W_out = nn.Linear(width, 3 * n_mixtures)
        self.n_mixtures = n_mixtures

        # Print model size
        self.num_params()

    def set_condition(self, c):
        if self.c_time is None:
            self.c_time = self.cond_time(c)
        if self.c_freq is None:
            self.c_freq = self.cond_freq(c)

    def forward(self, x, entries, flag_lasts, c=None):
        # x: [B, T, M]
        # Shift the inputs right for time-delay inputs
        x_time = F.pad(x, [0, 0, 1, -1, 0, 0])
        if self.central:
            x_central = self.central_input(x_time)
        else:
            x_central = None

        # x_time: [B, T, M, 1]
        x_time = x_time.unsqueeze(-1)
        x_time = self.time_input(x_time)

        # Shift the inputs up for freq-delay inputs
        x_freq = F.pad(x, [0, 0, 0, 0, 1, -1]).unsqueeze(-1)
        x_freq = self.freq_input(x_freq)

        if self.cond:
            assert c is not None
            self.set_condition(c)
            x_freq = x_freq + self.c_freq
            x_time = x_time + self.c_time

        # Run through the layers
        for layer in self.layers:
            x_time, x_central, x_freq = layer(x_time, x_central, x_freq, entries, flag_lasts)

        # Get the mixture params
        B, T, M, D = x_freq.size()
        x = self.W_out(x_freq)
        x = x.view(B, T, M, -1, 3)

        mu = x[:, :, :, :, 0]
        sigma = torch.exp(x[:, :, :, :, 1])
        pi = F.log_softmax(x[:, :, :, :, 2], dim=3)

        return mu, sigma, pi

    def num_params(self):
        parameters = filter(lambda p: p[1].requires_grad, self.parameters())
        parameters = sum(np.prod(p.size()) for p in parameters) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)
