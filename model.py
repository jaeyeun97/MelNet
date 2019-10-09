import torch

from functools import partial
from itertools import chain
from torch.optim.lr_scheduler import LambdaLR
# from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from datetime import datetime

from network import MelNet, FeatureExtraction
from utils import generate_splits, interleave, mdn_loss, get_grad_info, clip_grad


def get_model_fn(config):
    # pre-apply everything except device
    return partial(MelNetModel, config.mixtures, config.width, config.n_layers,
                   config.n_mels, config.timesteps,
                   mode=config.mode, lr_decay=config.lr_decay,
                   optimizer_cls=config.optimizer, optim_args=config.optim_args,
                   log_grad=config.log_grad, grad_clip=config.grad_clip, grad_scale=config.grad_scale,
                   amp_enable=config.amp_enable, amp_level=config.amp_level)


class MelNetModel(object):

    def __init__(self, mixtures, width, n_layers, n_mels, timesteps,
                 device, mode='train', lr_decay=False,
                 optimizer_cls=torch.optim.RMSprop, optim_args={},
                 log_grad=False, grad_clip=0, grad_scale=0,
                 amp_enable=False, amp_level='O1'):
        # Store each network in main memory
        self.n_layers = n_layers
        self.scale_count = len(self.n_layers) - 1
        self.n_mels = n_mels
        self.timesteps = timesteps
        self.device = device
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
        melnets = [MelNet(dims, self.n_layers[0],
                          n_mixtures=n_mixtures,
                          hook=hook).to(device=device)]
        f_exts = []
        # Upsample Networks
        for n_layer in self.n_layers[1:]:
            melnets.append(MelNet(dims, n_layer,
                                  n_mixtures=n_mixtures,
                                  cond=True, cond_dims=dims*4,
                                  hook=hook).to(device=device))
            f_exts.append(FeatureExtraction(dims, hook=hook).to(device=device))

        # Initialize Optimizers
        if self.mode == 'train':
            fext_optims = [optimizer_cls(model.parameters(), **optim_args)
                           for model in f_exts]
            melnet_optims = [optimizer_cls(model.parameters(), **optim_args)
                             for model in melnets]

            models, optims = amp.initialize(melnets + f_exts,
                                            melnet_optims + fext_optims,
                                            enabled=amp_enable,
                                            opt_level=amp_level)
            self.melnet_optims = optims[:len(self.n_layers)]
            self.fext_optims = optims[:len(self.n_layers)]

            if self.lr_decay:
                self.fext_scheds = [LambdaLR(optim, sched_fn)
                                    for optim in self.fext_optims]
                self.melnet_scheds = [LambdaLR(optim, sched_fn)
                                      for optim in self.melnet_optims]
        else:
            models, _ = amp.initialize(melnets + f_exts, opt_level="O1")

        melnets = models[:len(self.n_layers)]
        f_exts = models[len(self.n_layers):]
        self.melnets = [DDP(net, delay_allreduce=True) for net in melnets]  # , device_ids=[device]
        self.f_exts = [DDP(net, delay_allreduce=True) for net in f_exts]  # , device_ids=[device]

    def train(self):
        for net in chain(self.melnets, self.f_exts):
            net.train()

    def eval(self):
        for net in chain(self.melnets, self.f_exts):
            net.eval()

    def step(self, x, mode='train'):
        x = x.to(device=self.device)

        splits = generate_splits(x, self.scale_count)
        losses = []
        grad_infos = []

        # Run Each Network in reverse;
        # need to do it the right way when sampling
        # theoretically these can be parallelised
        for i in reversed(range(len(self.n_layers))):
            if mode == 'train':
                self.melnet_optims[i].zero_grad()
                if i != 0:
                    self.fext_optims[i-1].zero_grad()

            if i == 0:
                x = next(splits)
                cond = None
            else:
                cond, x = next(splits)
                cond = self.f_exts[i-1](cond)

            if cond is not None and torch.isnan(cond).any():
                print('NaN at FExt')

            mu, sigma, pi = self.melnets[i](x, cond)

            loss = mdn_loss(mu, sigma, pi, x)

            if mode == 'train':
                optimizers = self.melnet_optims[i] if i == 0 else \
                             [self.melnet_optims[i], self.fext_optims[i-1]]

                with amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()

                # Gradient Clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(amp.master_params(self.melnet_optims[i]), self.grad_clip)
                    if i != 0:
                        torch.nn.utils.clip_grad_value_(amp.master_params(self.fext_optims[i-1]), self.grad_clip)

                if self.grad_scale > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.melnet_optims[i]), self.grad_scale)
                    if i != 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.fext_optims[i-1]), self.grad_scale)

                self.melnet_optims[i].step()
                if i != 0:
                    self.fext_optims[i-1].step()

                if self.lr_decay:
                    self.melnet_scheds[i].step()
                    if i != 0:
                        self.fext_scheds[i-1].step()

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
            melnet = self.melnets[i].module

            if output is not None:
                f_ext = self.f_exts[i - 1].module
                cond = f_ext(output)
            else:
                cond = None

            # Autoregression
            t = datetime.now()

            for j in range(timesteps):
                for k in range(num_mels):
                    torch.cuda.synchronize()
                    mu, sigma, pi = (item[:, j, k].float() for item in melnet(x.clone(), cond))
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
                data[f'fext_optim_{i}'] = self.fext_optims[i - 1].state_dict()
                if self.lr_decay:
                    data[f'fext_sched_{i}'] = self.fext_scheds[i - 1].state_dict()
            data[f'melnet_{i}'] = self.melnets[i].state_dict()
            data[f'melnet_optim_{i}'] = self.melnet_optims[i].state_dict()
            if self.lr_decay:
                data[f'melnet_sched_{i}'] = self.melnet_scheds[i].state_dict()
            data['amp'] = amp.state_dict()
        return data

    def load_networks(self, checkpoint):
        if checkpoint is None:
            return

        for i in range(len(self.n_layers)):
            if i != 0:
                self.f_exts[i - 1].load_state_dict(checkpoint[f'f_ext_{i}'])
            self.melnets[i].load_state_dict(checkpoint[f'melnet_{i}'])
            if self.mode == 'train':
                self.melnet_optims[i].load_state_dict(checkpoint[f'melnet_optim_{i}'])
                state_dict_to_device(self.melnet_optims[i], self.device)

                if i != 0:
                    self.fext_optims[i-1].load_state_dict(checkpoint[f'fext_optim_{i}'])
                    state_dict_to_device(self.fext_optims[i-1], self.device)

                if f'melnet_sched_{i}' in checkpoint:
                    self.melnet_scheds[i].load_state_dict(checkpoint[f'melnet_sched_{i}'])

                if f'fext_sched_{i}' in checkpoint:
                    self.fext_scheds[i-1].load_state_dict(checkpoint[f'fext_sched_{i}'])

                if 'amp' in checkpoint:
                    amp.load_state_dict(checkpoint['amp'])


def state_dict_to_device(module, device):
    for state in module.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def sched_fn(x):
    return 0.1 ** (x ** 0.4 / 50)
