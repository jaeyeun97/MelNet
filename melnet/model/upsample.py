import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UpsampleLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # Is this needed? Might we just take the dimensional output of the previous network?
        self.freq_rnn = nn.GRU(dims, dims, batch_first=True, bidirectional=True)
        self.time_rnn = nn.GRU(dims, dims, batch_first=True, bidirectional=True)
        self.W_out = nn.Linear(4 * dims, dims)

    def forward(self, x):
        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x.size()

        # Collapse the first two axes
        x_time = x.transpose(1, 2).contiguous().view(-1, T, D)  # [B*M, T, D]
        x_freq = x.view(-1, M, D)  # [B*T, M, D]

        # Run through the rnns, do not use hidden states
        x_time, _ = self.time_rnn(x_time)
        x_freq, _ = self.freq_rnn(x_freq)

        # Reshape the first two axes back to original
        x_time = x_time.view(B, M, T, 2 * D).transpose(1, 2)
        x_freq = x_freq.view(B, T, M, 2 * D)

        # And concatenate for output
        x = x + self.W_out(torch.cat([x_time, x_freq], dim=3))
        return x

class UpsampleTier(nn.Module):
    """This is the non autoregressive version, which is different than the version on the paper."""
    def __init__(self, dims, n_layers, n_mixtures):
        super().__init__()
        self.W_in = nn.Linear(1, dims)
        self.layers = nn.ModuleList([UpsampleLayer(dims) for _ in range(n_layers)])
        self.W_out = nn.Linear(dims, 3 * n_mixtures)
        self.num_params()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.W_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.W_out(x)

        B, T, M, D = x.size()
        x = x.view(B, T, M, -1, 3)

        mu = x[:, :, :, :, 0]
        sigma = torch.exp(x[:, :, :, :, 1])
        pi = F.log_softmax(x[:, :, :, :, 2], dim=3)

        return mu, sigma, pi

    def num_params(self):
        parameters = filter(lambda p: p[1].requires_grad, self.parameters())
        parameters = sum(np.prod(p.size()) for p in parameters) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)
