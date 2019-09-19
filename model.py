import torch

from itertools import chain
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, get_grad_info, clip_grad


class MelNetModel(object):

    def __init__(self, config):
        # Store each network in main memory
        self.config = config
        self.n_layers = self.config.n_layers
        self.scale_count = len(self.n_layers) - 1
        self.device = self.config.device
        self.dtype = self.config.dtype
        self.preprocess = None

        dims = self.config.width
        n_mixtures = self.config.mixtures
        if config.grad_clip > 0:
            hook = clip_grad(config.grad_clip)
        else:
            hook = None

        # Unconditional first
        self.melnets = [MelNet(dims, self.n_layers[0],
                               n_mixtures=n_mixtures,
                               hook=hook).to(self.dtype)]
        self.f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            self.melnets.append(MelNet(dims, n_layer,
                                       n_mixtures=n_mixtures,
                                       cond=True, cond_dims=dims*4,
                                       hook=hook).to(self.dtype))
            self.f_exts.append(FeatureExtraction(dims, hook=hook).to(self.dtype))

        # Initialize Optimizers
        if self.config.mode == 'train':
            self.schedulers = []
            self.optimizers = []
            for i in reversed(range(len(self.n_layers))):
                melnet = self.melnets[i]
                if i != 0:
                    f_ext = self.f_exts[i-1]
                it = melnet.parameters() if i == 0 else chain(f_ext.parameters(), melnet.parameters())
                self.optimizers.insert(0, self.config.optimizer(it, **self.config.optim_args))
                if self.config.lr_decay:
                    print("Using Decay")
                    self.schedulers.insert(0, LambdaLR(self.optimizers[0], lambda x: 0.1 ** (x ** 0.4 / 50)))

    def train(self):
        for net in chain(self.melnets, self.f_exts):
            net.train()

    def eval(self):
        for net in chain(self.melnets, self.f_exts):
            net.eval()

    def set_preprocess(self, preprocess):
        self.preprocess = preprocess

    def step(self, x, mode='train'):
        x = x.to(dtype=self.dtype, device=self.device)

        if self.preprocess is not None:
            x = self.preprocess(x)

        splits = generate_splits(x, self.scale_count)
        losses = []
        grad_infos = []

        # Run Each Network in reverse;
        # need to do it the right way when sampling
        # theoretically these can be parallelised
        for i in reversed(range(len(self.n_layers))):
            if mode == 'train':
                self.optimizers[i].zero_grad()

            melnet = self.melnets[i].to(self.device)

            if i == 0:
                x = next(splits)
            else:
                f_ext = self.f_exts[i - 1].to(self.device)
                cond, x = next(splits)
                melnet.set_condition(f_ext(cond))
            mu, sigma, pi = melnet(x)

            loss = mdn_loss(mu, sigma, pi, x)

            if mode == 'train':
                loss.backward()

                # Gradient Clipping
                if self.config.grad_clip > 0:
                    if i == 0:
                        it = melnet.parameters()
                    else:
                        it = chain(f_ext.parameters(), melnet.parameters())
                    torch.nn.utils.clip_grad_value_(it, self.config.grad_clip)

                if self.config.grad_scale > 0:
                    if i == 0:
                        it = melnet.parameters()
                    else:
                        it = chain(f_ext.parameters(), melnet.parameters())
                    torch.nn.utils.clip_grad_norm_(it, self.config.grad_scale)
                self.optimizers[i].step()
                if self.config.optimizer == torch.optim.Adam:
                    self.schedulers[i].step()

                # Gradient logging
                if i == 0:
                    grad_info = get_grad_info(melnet)
                else:
                    grad_info = get_grad_info(f_ext, melnet)
                grad_infos.insert(0, grad_info)

            self.melnets[i] = melnet  # .cpu()
            if i != 0:
                self.f_exts[i-1] = f_ext  # .cpu()
            losses.insert(0, loss.item())

        return losses, grad_infos

    def sample(self):
        div_factor = (2 ** (self.scale_count // 2))
        num_mels = self.config.n_mels // div_factor
        timesteps = self.config.timesteps // div_factor

        if self.scale_count % 2 != 0:
            num_mels //= 2

        axis = False
        output = None
        for i in range(len(self.n_layers)):
            x = torch.zeros(1, timesteps, num_mels).to(device=self.device)
            melnet = self.melnets[i].to(self.device)

            if output is not None:
                print(output)
                f_ext = self.f_exts[i - 1].to(self.device)
                melnet.set_condition(f_ext(output))

            # Autoregression
            t = datetime.now()
            try:
                for j in range(timesteps):
                    for k in range(num_mels):
                        mu, sigma, pi = (item[0, j, k] for item in melnet(x))
                        idx = pi.exp().multinomial(1).item()
                        x[0, j, k] = torch.normal(mu, sigma)[idx]
                        x = x.clone()
                print(f"Sampling Time: {datetime.now() - t}")
            except RuntimeError:
                __import__('ipdb').set_trace()

            if i == 0:
                output = x
            else:
                output = interleave(output, x, axis)
                _, timesteps, num_mels = output.size()
                axis = not axis

        return output

    def save_networks(self):
        data = dict()
        for i in range(len(self.n_layers)):
            if i != 0:
                data[f'f_ext_{i}'] = self.f_exts[i - 1].state_dict()
            data[f'melnet_{i}'] = self.melnets[i].state_dict()
            data[f'optimizer_{i}'] = self.optimizers[i].state_dict()
        return data

    def load_networks(self, checkpoint):
        if checkpoint is None:
            return
        for i in range(len(self.n_layers)):
            if i != 0:
                self.f_exts[i - 1].load_state_dict(checkpoint[f'f_ext_{i}'])
            self.melnets[i].load_state_dict(checkpoint[f'melnet_{i}'])
            if self.config.mode == 'train':
                self.optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}'])

                for state in self.optimizers[i].state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)


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
