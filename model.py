import torch
import torch.nn as nn

from itertools import chain

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, sample


class MelNetModel(object):

    n_layers = [12, 5, 4, 3, 2, 2]
    timestep = 80
    num_mels = 64

    def __init__(self, config):
        # Store each network in main memory
        self.config = config
        self.device = self.config.device

        dims = self.config.width
        n_mixtures = self.config.mixtures

        # Unconditional first
        self.melnets = [MelNet(dims, self.n_layers[0], n_mixtures=n_mixtures)]
        self.f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            self.melnets.append(MelNet(dims, n_layer, n_mixtures=n_mixtures, cond=True, cond_dims=dims*4))
            self.f_exts.append(FeatureExtraction(dims))

        if self.config.is_train:
            self.optimizers = []
            for i in reversed(range(len(self.n_layers))):
                melnet = self.melnets[i]
                f_ext = self.f_exts[i-1]

                melnet.train()
                if i != 0:
                    f_ext.train()

                # set optimizer parameters
                self.optimizers.insert(0, self.config.optimizer(chain(melnet.parameters(), f_ext.parameters()), lr=self.config.lr))
        else:
            for net in chain(self.melnets, self.f_exts):
                net.eval()

    def step(self, x):
        x = x.to(self.device)
        x = self.config.preprocess(x)
        # TODO: Check and fix network instead
        x = x.transpose(1, 2)

        l = len(self.n_layers)
        splits = generate_splits(x, l)

        # Run Each Network in reverse;
        # need to do it the right way when sampling
        # theoretically these can be parallelised
        for i in reversed(range(l)):
            if self.config.is_train:
                self.optimizers[i].zero_grad()

            melnet = self.melnets[i].to(self.device)

            if i == 0:
                x = next(splits)
                mu, sigma, pi = melnet(x)
            else:
                f_ext = self.f_exts[i - 1].to(self.device)
                cond, x = next(splits)
                features = f_ext(cond)
                mu, sigma, pi = melnet(x, features)

            loss = mdn_loss(mu, sigma, pi, x)
            if self.config.is_train:
                loss.backward()
                self.optimizers[i].step()

            # re-move network to cpu
            self.melnets[i] = melnet.cpu()

            # TODO: store network and report loss

    def sample(self):
        cond = None
        melnet = self.melnets[0].to(self.device)
        timesteps = self.timesteps
        num_mels = self.num_mels
        for i in range(len(self.n_layers)):
            x = torch.zeros(1, timesteps, num_mels).to(self.device)
            melnet = self.melnets[i].to(self.device)
            
            for _ in range(timesteps * num_mels):
                mu, sigma, pi = melnet(x, cond)
                x = sample(mu, sigma, pi)
            if i == 0:
                cond = x
            else:
                cond = interleave(cond, x)
                _, timesteps, num_mels = cond.size()
