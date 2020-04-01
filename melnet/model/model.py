import torch
import torch.nn as nn

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from .initial import InitialTier
from .upsample import UpsampleTier
from .utils import generate_splits, interleave, mdn_loss, get_div_factors, sample_mdn


class MelNet(nn.Module):
    def __init__(self, width, n_freq, n_layers, n_mixtures, mode='train',
                 optimizer_cls=torch.optim.RMSprop, optim_args={'lr': 0.0001},
                 grad_acc=8, amp_level='O1', distributed=False):
        super().__init__()
        self.grad_acc = grad_acc
        self.counts = len(n_layers)
        self.t_div, self.f_div = get_div_factors(self.counts)
        self.n_initial_freq = n_freq // (2 ** self.f_div)

        self.tiers = [
            InitialTier(width, self.n_initial_freq, n_layers[0], n_mixtures).cuda(),
            *(UpsampleTier(width, l, n_mixtures).cuda() for l in n_layers[1:])
        ]

        if mode == 'train':
            self.optimizers = [optimizer_cls(t.parameters(), **optim_args) for t in self.tiers]
            self.train()
        else:
            self.optimizers = None
            self.eval()

        self.tiers, self.optimizers = amp.initialize(self.tiers,
                                                     optimizers=self.optimizers,
                                                     opt_level=amp_level)

        if distributed:
            self.tiers = [DDP(t, delay_allreduce=True) for t in self.tiers]
        self.tiers = nn.ModuleList(self.tiers)

        self.streams = [torch.cuda.Stream() for _ in range(self.counts)]

    def sync_streams(self):
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)

    def zero_grad(self):
        if self.optimizers:
            for i in range(self.counts):
                with torch.cuda.stream(self.streams[i]):
                    self.optimizers[i].zero_grad()

    def step(self):
        if self.optimizers:
            for i in range(self.counts):
                with torch.cuda.stream(self.streams[i]):
                    self.optimizers[i].step()

    def optimizer_state_dict(self):
        if self.optimizers:
            return [optim.state_dict() for optim in self.optimizers]

    def optimizer_load_state_dict(self, state_dicts):
        if self.optimizers:
            for i in range(self.counts):
                with torch.cuda.stream(self.streams[i]):
                    self.optimizers[i].load_state_dict(state_dicts[i])
                    for state in self.optimizers[i].state.values():
                        for k, v in state.items():
                            state[k] = v.cuda()

    def forward(self, x, entries, flag_lasts, step_iter=False):
        splits = generate_splits(x, self.counts)
        losses = []

        for i in range(self.counts):
            with torch.cuda.stream(self.streams[i]):
                if i == 0:
                    curr = splits[i]
                    mu, sigma, pi = self.tiers[i](curr, entries, flag_lasts)
                else:
                    prev, curr = splits[i]
                    mu, sigma, pi = self.tiers[i](prev)

                loss = mdn_loss(mu, sigma, pi, curr) / self.grad_acc

                if self.training:
                    with amp.scale_loss(loss, self.optimizers[i],
                                        delay_unscale=not step_iter) as scaled_loss:
                        scaled_loss.backward()

                    if step_iter:
                        self.optimizers[i].step()
                        self.optimizers[i].zero_grad()

                losses.append(loss.clone().detach())

        return torch.stack(losses)

    def sample(self, timesteps):
        self.eval()
        timesteps //= (2 ** self.t_div)

        entries = [-1]
        x = torch.zeros(1, 1, self.n_initial_freq).cuda()
        for t in range(timesteps):
            flag_lasts = [t == timesteps - 1]
            for f in range(self.n_initial_freq):
                mu, sigma, pi = self.tiers[0](x, entries, flag_lasts)
                x[:, t, f] = sample_mdn(mu, sigma, pi)[:, t, f]
            if t != timesteps - 1:
                x = torch.cat([x, torch.zeros(1, 1, self.n_initial_freq).cuda()], 1)

        axis = True
        for tier in self.tiers[1:]:
            mu, sigma, pi = tier(x)
            y = sample_mdn(mu, sigma, pi)
            x = interleave(x, y, axis)
            axis = not axis

        return x
