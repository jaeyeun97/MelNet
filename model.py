import torch
import torch.nn as nn

from itertools import chain

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, sample
from audio import MelScale, Spectrogram
from torchvision.transforms import Compose


class MelNetModel(object):

    n_layers = [12, 5, 4, 3, 2, 2]
    timesteps = 80
    num_mels = 64
    scale_count = len(n_layers) - 2

    def __init__(self, config):
        # Store each network in main memory
        self.config = config
        self.device = self.config.device
        self.dtype = self.config.dtype

        dims = self.config.width
        n_mixtures = self.config.mixtures
        self.setup_preprocess()

        # Unconditional first
        self.melnets = [MelNet(dims, self.n_layers[0], n_mixtures=n_mixtures).to(self.dtype)]
        self.f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            self.melnets.append(MelNet(dims, n_layer, n_mixtures=n_mixtures, cond=True, cond_dims=dims*4).to(self.dtype))
            self.f_exts.append(FeatureExtraction(dims).to(self.dtype))

        if self.config.mode == 'train':
            self.optimizers = []
            for i in reversed(range(len(self.n_layers))):
                melnet = self.melnets[i]  
                melnet.train()

                if i != 0:
                    f_ext = self.f_exts[i-1]
                    f_ext.train()

                it = melnet.parameters() if i == 0 else chain(melnet.parameters(), f_ext.parameters())
                self.optimizers.insert(0, self.config.optimizer(it, lr=self.config.lr))
        else:
            for net in chain(self.melnets, self.f_exts):
                net.eval()

    def setup_preprocess(self):
        self.preprocess = Compose([
            Spectrogram(n_fft=self.config['n_fft'],
                        win_length=self.config['win_length'],
                        hop_length=self.config['hop_length'],
                        normalized=True,
                        dtype=self.dtype,
                        device=self.device),
            MelScale(sample_rate=self.config['sample_rate'],
                     n_fft=self.config['n_fft'],
                     n_mels=self.config['n_mels'],
                     dtype=self.dtype,
                     device=self.device)
        ])

    def step(self, x):
        x = x.to(dtype=self.dtype, device=self.device)
        x = self.preprocess(x)

        splits = generate_splits(x, self.scale_count)

        # Run Each Network in reverse;
        # need to do it the right way when sampling
        # theoretically these can be parallelised
        for i in reversed(range(len(self.n_layers))):
            if self.config.mode == 'train':
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

            if self.config.mode == 'train':
                loss.backward()
                self.optimizers[i].step()

            # re-move network to cpu
            self.melnets[i] = melnet.cpu()

            # TODO: store network and report loss
            print(loss)

    def sample(self):
        cond = None
        melnet = self.melnets[0].to(self.device)
        timesteps = self.timesteps
        num_mels = self.num_mels
        axis = False
        for i in range(len(self.n_layers)):
            x = torch.zeros(1, timesteps, num_mels).to(self.device)
            melnet = self.melnets[i].to(self.device)

            # Autoregression
            for _ in range(timesteps * num_mels):
                mu, sigma, pi = melnet(x, cond)
                x = sample(mu, sigma, pi)

            if i == 0:
                cond = x
            else:
                # One extra cond generated
                cond = interleave(cond, x, axis)
                _, timesteps, num_mels = cond.size()
                axis = not axis
        return x


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = MelNetModel(config) 

    import librosa
    x, sr = librosa.load(librosa.util.example_audio_file(),
                         sr=config.sample_rate,
                         duration=config.hop_length * 319 / config.sample_rate)
    x = torch.from_numpy(x).unsqueeze(0)
    for _ in range(20):
        model.step(x)
