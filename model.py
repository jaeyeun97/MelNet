import os
import torch

from itertools import chain
from datetime import datetime

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, sample


class MelNetModel(object):

    n_layers = [12, 5, 4, 3, 2, 2]
    scale_count = len(n_layers) - 2

    def __init__(self, config):
        # Store each network in main memory
        self.config = config
        self.device = self.config.device
        self.dtype = self.config.dtype
        self.preprocess = None

        dims = self.config.width
        n_mixtures = self.config.mixtures

        # Unconditional first
        self.melnets = [MelNet(dims, self.n_layers[0], n_mixtures=n_mixtures).to(self.dtype)]
        self.f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            self.melnets.append(MelNet(dims, n_layer, n_mixtures=n_mixtures, cond=True, cond_dims=dims*4).to(self.dtype))
            self.f_exts.append(FeatureExtraction(dims).to(self.dtype))

        # Initialize Optimizers
        if self.config.mode == 'train':
            self.optimizers = []
            for i in reversed(range(len(self.n_layers))):
                melnet = self.melnets[i]
                if i != 0:
                    f_ext = self.f_exts[i-1]
                it = melnet.parameters() if i == 0 else chain(f_ext.parameters(), melnet.parameters())
                self.optimizers.insert(0, self.config.optimizer(it, lr=self.config.lr))  # , momentum=0.9)) 

    def train(self):
        for net in chain(self.melnets, self.f_exts):
            net.train()

    def eval(self):
        for net in chain(self.melnets, self.f_exts):
            net.eval()

    def set_preprocess(self, preprocess):
        self.preprocess = preprocess

    def step(self, x):
        x = x.to(dtype=self.dtype, device=self.device)

        if self.preprocess is not None:
            x = self.preprocess(x)

        splits = generate_splits(x, self.scale_count)
        losses = []

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

            # Gradient clipping
            loss = mdn_loss(mu, sigma, pi, x)

            if self.config.mode == 'train':
                loss.backward()
                it = melnet.parameters() if i == 0 else chain(f_ext.parameters(), melnet.parameters())
                torch.nn.utils.clip_grad_norm_(it, self.config.grad_clip)
                self.optimizers[i].step()

            # re-move network to cpu
            self.melnets[i] = melnet.cpu()
            losses.append(float(loss))

        return tuple(losses)

    def sample(self):
        cond = None
        melnet = self.melnets[0].to(self.device)
        timesteps = self.config.n_mel * 2 // self.scale_count
        num_mels =  self.config.timesteps * 2 // self.scale_count
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

    def save_networks(self, epoch=0, it=0):
        now = datetime.now()
        timestamp = datetime.timestamp(now)

        if it != 0:
            name = f'iter_{it}'
        elif epoch != 0:
            name = f'epoch_{epoch}'
        else:
            name = f'time_{int(timestamp)}'

        data = {'epoch': epoch, 'timestamp': timestamp}
        for i in range(len(self.n_layers)):
            if i != 0:
                data[f'f_ext_{i}'] = self.f_exts[i - 1].state_dict()
            data[f'melnet_{i}'] = self.melnets[i].state_dict()
            data[f'optimizer_{i}'] = self.optimizers[i].state_dict()
        torch.save(data, os.path.join(self.config.checkpoint_dir, f'{name}.pth'))

    def load_networks(self, epoch=0, it=0, timestamp=0):
        if it != 0:
            name = f'iter_{it}'
        elif epoch != 0:
            name = f'epoch_{epoch}'
        else:
            name = f'time_{timestamp}'

        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, f'{name}.pth'))
        for i in range(len(self.n_layers)):
            if i != 0:
                self.f_exts[i - 1].load_state_dict(checkpoint[f'f_ext_{i}'])
            self.melnets[i].load_state_dict(checkpoint[f'melnet_{i}'])
            self.optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}'])

        epoch = checkpoint['epoch']
        time = str(datetime.fromtimestamp(checkpoint['timestamp']))
        print(f"Loading Network from Epoch {epoch} at {time}")


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = MelNetModel(config)

    import librosa
    x, sr = librosa.load(librosa.util.example_audio_file(),
                         sr=config.sample_rate,
                         duration=config.frame_length / config.sample_rate)
    x = torch.from_numpy(x).unsqueeze(0)
    print(x.size())
    for _ in range(20):
        model.step(x)
