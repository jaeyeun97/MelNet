import torch
import numpy as np

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def test(model_fn, dataset, batch_size, q, rank, world_size, device,
         checkpoint=None, num_workers=0, shuffle=False):
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    model = model_fn(device)
    model.load_networks(checkpoint)
    model.eval()
    sampler = DistributedSampler(dataset,
                                 num_replicas=world_size,
                                 rank=rank)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers // world_size,
                        pin_memory=True,
                        sampler=sampler)

    with torch.no_grad():
        losses = [model.step(x, 'test')[0] for x in loader]

    q.put(np.mean(losses, axis=0))
