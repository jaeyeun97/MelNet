import torch

from functools import partial
from itertools import chain
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, get_grad_info, clip_grad


def get_model_fn(config):
    # pre-apply everything except device
    return partial(MelNetModel, config.mixtures, config.width, config.n_layers,
                   config.n_mels, config.timesteps, config.dtype,
                   mode=config.mode, lr_decay=config.lr_decay,
                   optimizer_cls=config.optimizer, optim_args=config.optim_args,
                   log_grad=config.log_grad, grad_clip=config.grad_clip, grad_scale=config.grad_scale)


class MelNetModel(object):

    def __init__(self, mixtures, width, n_layers, n_mels, timesteps,
                 dtype, device, mode='train', lr_decay=False,
                 optimizer_cls=torch.optim.RMSprop, optim_args={},
                 log_grad=False, grad_clip=0, grad_scale=0):
        # Store each network in main memory
        self.n_layers = n_layers
        self.scale_count = len(self.n_layers) - 1
        self.n_mels = n_mels
        self.timesteps = timesteps
        self.device = device
        self.dtype = dtype
        self.mode = mode

        self.lr_decay = lr_decay
        self.log_grad = log_grad
        self.grad_clip = grad_clip
        self.grad_scale = grad_scale

        dims = width
        n_mixtures = mixtures

        if grad_clip > 0:
            hook = clip_grad(grad_clip)
        else:
            hook = None

        # Unconditional first
        self.melnets = [MelNet(dims, self.n_layers[0],
                               n_mixtures=n_mixtures,
                               hook=hook).to(dtype=self.dtype, device=device)]
        self.f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            self.melnets.append(MelNet(dims, n_layer,
                                       n_mixtures=n_mixtures,
                                       cond=True, cond_dims=dims*4,
                                       hook=hook).to(dtype=dtype, device=device))
            self.f_exts.append(FeatureExtraction(dims, hook=hook).to(dtype=dtype, device=device))

        self.melnets = [DDP(net, device_ids=[device]) for net in self.melnets]
        self.f_exts = [DDP(net, device_ids=[device]) for net in self.f_exts]

        # Initialize Optimizers
        if self.mode == 'train':
            self.schedulers = []
            self.optimizers = []
            for i in reversed(range(len(self.n_layers))):
                melnet = self.melnets[i]
                if i != 0:
                    f_ext = self.f_exts[i-1]
                it = melnet.parameters() if i == 0 else chain(f_ext.parameters(), melnet.parameters())
                self.optimizers.insert(0, optimizer_cls(it, **optim_args))
                if self.lr_decay:
                    self.schedulers.insert(0, LambdaLR(self.optimizers[0], lambda x: 0.1 ** (x ** 0.4 / 50))) 

    def train(self):
        for net in chain(self.melnets, self.f_exts):
            net.train()

    def eval(self):
        for net in chain(self.melnets, self.f_exts):
            net.eval()

    def step(self, x, mode='train'):
        x = x.to(dtype=self.dtype, device=self.device)

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
                if self.grad_clip > 0:
                    if i == 0:
                        it = self.melnets[0].parameters()
                    else:
                        it = chain(self.f_exts[i-1].parameters(), self.melnets[i].parameters())
                    torch.nn.utils.clip_grad_value_(it, self.grad_clip)

                if self.grad_scale > 0:
                    if i == 0:
                        it = self.melnets[0].parameters()
                    else:
                        it = chain(self.f_exts[i-1].parameters(), self.melnets[i].parameters())
                    torch.nn.utils.clip_grad_norm_(it, self.grad_scale)

                self.optimizers[i].step()

                if self.lr_decay:
                    self.schedulers[i].step()

                # Gradient logging
                if self.log_grad:
                    if i == 0:
                        grad_info = get_grad_info(self.melnets[i])
                    else:
                        grad_info = get_grad_info(self.f_exts[i-1], self.melnets[i])
                    grad_infos.insert(0, grad_info)
            losses.insert(0, loss.item())

        return losses, grad_infos

    def sample(self):
        div_factor = (2 ** (self.scale_count // 2))
        num_mels = self.n_mels // div_factor
        timesteps = self.timesteps // div_factor
        batchsize = 1

        if self.scale_count % 2 != 0:
            num_mels //= 2

        axis = False
        output = None
        for i in range(len(self.n_layers)):
            x = torch.zeros(batchsize, timesteps, num_mels).to(self.device)
            melnet = self.melnets[i]

            if output is not None:
                f_ext = self.f_exts[i - 1]
                cond = f_ext.module(output)
            else:
                cond = None

            # Autoregression
            t = datetime.now()

            for j in range(timesteps):
                for k in range(num_mels):
                    torch.cuda.synchronize()
                    mu, sigma, pi = (item[:, j, k] for item in melnet.module(x.clone(), cond))
                    idx = pi.exp().multinomial(1)
                    x[:, j, k] = torch.normal(mu, sigma).gather(-1, idx).squeeze(-1)
            print(f"Sampling Time: {datetime.now() - t}")

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
            if self.lr_decay:
                data[f'scheduler_{i}'] = self.schedulers[i].state_dict()
        return data

    def load_networks(self, checkpoint):
        if checkpoint is None:
            return
        for i in range(len(self.n_layers)):
            if i != 0:
                self.f_exts[i - 1].load_state_dict(checkpoint[f'f_ext_{i}'])
            self.melnets[i].load_state_dict(checkpoint[f'melnet_{i}'])
            if self.mode == 'train':
                self.optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}'])
                if f'scheduler_{i}' in checkpoint:
                    self.schedulers[i].load_state_dict(checkpoint[f'scheduler_{i}'])

                for state in self.optimizers[i].state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
