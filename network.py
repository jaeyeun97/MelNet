import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import Layer
from utils import clip_grad


class FeatureExtraction(nn.Module):
    def __init__(self, dims, hook):
        super().__init__()
        # Is this needed? Might we just take the dimensional output of the previous network?
        self.time_input = nn.Linear(1, dims)
        self.freq_input = nn.Linear(1, dims)

        self.freq_rnn = nn.GRU(dims, dims, batch_first=True, bidirectional=True)
        self.time_rnn = nn.GRU(dims, dims, batch_first=True, bidirectional=True)

        self.hook = hook

    def forward(self, x):
        x = x.unsqueeze(-1)

        x_time = self.time_input(x)
        x_freq = self.freq_input(x)

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x_time.size()

        # Collapse the first two axes
        x_time = x_time.transpose(1, 2).contiguous().view(-1, T, D)  # [B*M, T, D]
        x_freq = x_freq.view(-1, M, D)  # [B*T, M, D]

        # Run through the rnns
        x_time, _ = self.time_rnn(x_time)
        x_freq, _ = self.freq_rnn(x_freq)

        if self.hook:
            x_time.register_hook(self.hook)
            x_freq.register_hook(self.hook)

        # Reshape the first two axes back to original
        x_time = x_time.view(B, M, T, 2 * D).transpose(1, 2)
        x_freq = x_freq.view(B, T, M, 2 * D)

        # And concatenate for output
        x = torch.cat([x_time, x_freq], dim=3)

        # output: [B, T, M, 4 * D]
        return x


class MelNet(nn.Module):
    def __init__(self, dims, n_layers, n_mixtures=10, hook=None, cond=False, cond_dims=1):
        super().__init__()
        # Input layers
        self.freq_input = nn.Linear(1, dims)
        self.time_input = nn.Linear(1, dims)

        if cond:
            # Paper states that there are two condition networks: W^t_z, W^f_z
            self.cond_freq = nn.Linear(cond_dims, dims)
            self.cond_time = nn.Linear(cond_dims, dims)
            self.c_freq = None
            self.c_time = None
        self.cond = cond

        # Main layers
        self.layers = nn.Sequential(
            *[Layer(dims, hook) for _ in range(n_layers)]
        )

        # Output layer
        self.fc_out = nn.Linear(2 * dims, 3 * n_mixtures)
        self.n_mixtures = n_mixtures

        # Print model size
        self.num_params()

    def set_condition(self, c):
        if self.cond:
            self.c_time = self.cond_time(c)
            self.c_freq = self.cond_freq(c)

    def forward(self, x, c=None):
        # x: [B, T, M]
        # Shift the inputs right for time-delay inputs
        # x_time: [B, T, M, 1]
        x_time = F.pad(x, [0, 0, 1, -1, 0, 0]).unsqueeze(-1)
        # Shift the inputs up for freq-delay inputs
        x_freq = F.pad(x, [0, 0, 0, 0, 1, -1]).unsqueeze(-1)

        # Initial transform from 1 to dims
        # x_time : [B, T, M, D]
        x_time = self.time_input(x_time)
        # x_freq: [B, T, M, D]
        x_freq = self.freq_input(x_freq)

        if self.cond:
            if c is not None:
                self.set_condition(c)
            x_freq = x_freq + self.c_freq
            x_time = x_time + self.c_time

        # Run through the layers
        x = (x_time, x_freq)
        x_time, x_freq = self.layers(x)

        # Get the mixture params
        x = torch.cat([x_time, x_freq], dim=-1)
        x = self.fc_out(x)
        B, T, M, D = x.size()
        x = x.reshape(B, T, M, self.n_mixtures, 3)
        mu = x[:, :, :, :, 0]
        sigma = torch.exp(x[:, :, :, :, 1])
        pi = nn.functional.log_softmax(x[:, :, :, :, 2], dim=3)
        # pi = nn.functional.softmax(x[:, :, :, :, 2], dim=3)
        return mu, sigma, pi

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)


if __name__ == '__main__':
    batchsize = 1
    timesteps = 80
    num_mels = 64
    dims = 256
    n_layers = 2

    gpu = torch.device('cuda:0')
    model = MelNet(dims, n_layers, cond=True, cond_dims=dims*4).to(torch.half).to(gpu)
    f_ext = FeatureExtraction(dims).to(torch.half).to(gpu)

    x = torch.ones(batchsize, timesteps, num_mels, device=gpu, dtype=torch.half)
    c = torch.zeros(batchsize, timesteps, num_mels, device=gpu, dtype=torch.half)

    model.train()
    f_ext.train()
    print("Input Shape:", x.shape)
    c = f_ext(x)
    print("Conditional Shape", c.shape)
    mu, sigma, pi = model(x, c)
    print("Output Shape", mu.shape)

    from utils import mdn_loss, sample
    print(mdn_loss(mu, sigma, pi, x))
    print(sample(mu, sigma, pi).size())

    # 64 -> 64 -> 128 -> 128 -> 256 -> 256 # -> 512 -> 512 
    # 80 -> 80 -> 80  -> 160 -> 160 -> 320 # -> 320 -> 640
    # 12 -> 5  -> 4   -> 3   -> 2   -> 2
