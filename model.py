import torch
import torch.nn as nn

from itertools import chain
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import datetime

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, get_grad_info, clip_grad


class MelNetModel(object):

    def __init__(self, config):
        # Store each network in main memory
        self.config = config
        self.n_layers = self.config.n_layers
        self.scale_count = len(self.n_layers) - 1
        self.devices = self.config.devices
        self.dtype = self.config.dtype

        dims = self.config.width
        n_mixtures = self.config.mixtures
        if config.grad_clip > 0:
            hook = clip_grad(config.grad_clip)
        else:
            hook = None

        # Unconditional first
        self.melnets = [MelNet(dims, self.n_layers[0],
                               n_mixtures=n_mixtures,
                               hook=hook).to(dtype=self.dtype, device=self.devices[0])]
        self.f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            self.melnets.append(MelNet(dims, n_layer,
                                       n_mixtures=n_mixtures,
                                       cond=True, cond_dims=dims*4,
                                       hook=hook).to(dtype=self.dtype, device=self.devices[0]))
            self.f_exts.append(FeatureExtraction(dims, hook=hook).to(dtype=self.dtype, device=self.devices[0]))

        if len(self.devices) > 1:
            self.setup()
            self.melnets = [DDP(net) for net in self.melnets]
            self.f_exts = [DDP(net) for net in self.f_exts]

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

    def setup(self):
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", world_size=1, rank=0)
        torch.manual_seed(42)

    def train(self):
        for net in chain(self.melnets, self.f_exts):
            net.train()

    def eval(self):
        for net in chain(self.melnets, self.f_exts):
            net.eval()

    def step(self, x, mode='train'):
        x = x.to(dtype=self.dtype, device=self.devices[0])

        splits = generate_splits(x, self.scale_count)
        losses = []
        grad_infos = []

        # Run Each Network in reverse;
        # need to do it the right way when sampling
        # theoretically these can be parallelised
        for i in reversed(range(len(self.n_layers))):
            if mode == 'train':
                self.optimizers[i].zero_grad()

            if i == 0:
                x = next(splits)
                cond = None
            else:
                cond, x = next(splits)
                cond = self.f_exts[i-1](cond)

            mu, sigma, pi = self.melnets[i](x, cond)

            loss = mdn_loss(mu, sigma, pi, x)

            if mode == 'train':
                loss.backward()

                # Gradient Clipping
                if self.config.grad_clip > 0:
                    if i == 0:
                        it = self.melnets[0].parameters()
                    else:
                        it = chain(self.f_exts[i-1].parameters(), self.melnets[i].parameters())
                    torch.nn.utils.clip_grad_value_(it, self.config.grad_clip)

                if self.config.grad_scale > 0:
                    if i == 0:
                        it = self.melnets[0].parameters()
                    else:
                        it = chain(self.f_exts[i-1].parameters(), self.melnets[i].parameters())
                    torch.nn.utils.clip_grad_norm_(it, self.config.grad_scale)
                self.optimizers[i].step()
                if self.config.lr_decay:
                    self.schedulers[i].step()

                # Gradient logging
                if i == 0:
                    grad_info = get_grad_info(self.melnets[i])
                else:
                    grad_info = get_grad_info(self.f_exts[i-1], self.melnets[i])
                grad_infos.insert(0, grad_info)
            losses.insert(0, loss.item())

        return losses, grad_infos

    def sample(self):
        div_factor = (2 ** (self.scale_count // 2))
        num_mels = self.config.n_mels // div_factor
        timesteps = self.config.timesteps // div_factor
        batchsize = len(self.devices)

        if self.scale_count % 2 != 0:
            num_mels //= 2

        axis = False
        output = None
        for i in range(len(self.n_layers)):
            # TODO: change this somehow
            x = torch.zeros(batchsize, timesteps, num_mels)
            melnet = self.melnets[i]

            if output is not None:
                f_ext = self.f_exts[i - 1] # .to(self.device)
                cond = f_ext(output)
            else:
                cond = None

            # Autoregression
            t = datetime.now()
            try:
                for j in range(timesteps):
                    for k in range(num_mels):
                        torch.cuda.synchronize()
                        mu, sigma, pi = (item[:, j, k] for item in melnet(x.clone(), cond))
                        idx = pi.exp().multinomial(1)
                        x[:, j, k] = torch.normal(mu, sigma).gather(-1, idx).squeeze(-1)
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
            if self.config.lr_decay:
                data[f'scheduler_{i}'] = self.schedulers[i].state_dict()
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
                if f'scheduler_{i}' in checkpoint:
                    self.schedulers[i].load_state_dict(checkpoint[f'scheduler_{i}'])

                for state in self.optimizers[i].state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.devices[0])
