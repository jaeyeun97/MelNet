import os
import torch
import numpy as np
import time
import torch.distributed as dist

from functools import partial
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def get_train_fn(config, model_fn, train_dataset, val_dataset, checkpoint, postprocess):
    # pre-apply everything except rank, world_size, device, pipes
    return partial(train, model_fn, config.get_config(),
                   train_dataset, val_dataset, postprocess=postprocess,
                   checkpoint=checkpoint)


def train(model_fn, config, train_dataset, val_dataset, 
          rank, world_size, devices, pipes, postprocess=None,
          checkpoint=None):

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    device = devices[rank]
    pipes = pipes[rank]

    if checkpoint is not None:
        iteration = checkpoint['iteration'] + 1
        epoch = checkpoint['epoch'] + 1
    else:
        iteration = 1
        epoch = 1

    # Prepare Model
    model = model_fn(device)
    model.load_networks(checkpoint)
    model.train()
    dist.barrier()

    # Get Dataloader

    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=world_size,
                                       shuffle=config['shuffle'],
                                       rank=rank)
    val_sampler = DistributedSampler(val_dataset,
                                     num_replicas=world_size,
                                     shuffle=config['shuffle'],
                                     rank=rank)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              num_workers=config['num_workers'] // world_size,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'] // world_size,
                            pin_memory=True,
                            sampler=val_sampler)

    t = None
    while epoch < config['epochs'] + 1:
        # Minibatch
        for batch, x in enumerate(train_loader):
            # Timer
            if batch % config['time_interval'] == 0 and rank == 0:
                new_t = time.time()
                if t is not None:
                    print(f"Rank {rank}: time executed: {new_t - t}")
                t = new_t

            losses, grad_infos = model.step(x)

            # Logging Loss
            losses = torch.tensor(losses).to(device)
            dist.reduce(losses, 0)
            if rank == 0:
                losses = losses.div(world_size).cpu().numpy()
                pipes['train_loss'].send((iteration, losses))

            # Logging Gradient Flow
            if 'grad' in pipes and (iteration - 1) % 50 == 0 and rank == 0:
                pipes['grad'].send((iteration, grad_infos))

            # Save
            if iteration % config['iter_interval'] == 0 and rank == 0:
                print(f"Storing network for iteration {iteration}")
                save_model(model, epoch, iteration-1, config, True)

            # Validate
            if (iteration - 1) % config['val_interval'] == 0:
                with torch.no_grad():
                    model.eval()
                    losses = np.mean([model.step(x, 'validation')[0] for x in val_loader], axis=0)
                    losses = torch.from_numpy(losses).to(device)
                    dist.reduce(losses, 0)
                    if rank == 0:
                        losses = losses.div(world_size).cpu().numpy()
                        pipes['val_loss'].send((iteration, losses))
                model.train()
            iteration += 1

        # Store Network on epoch intervals
        if epoch % config['epoch_interval'] == 0 and rank == 0:
            print(f"Storing network for epoch {epoch}")
            save_model(model, epoch, iteration-1, config)

        # Sample
        if epoch % config['sample_interval'] == 0:
            print("Sampling..")
            with torch.no_grad():
                sample = model.sample()
                pipes['spectrogram'].send((iteration - 1, rank, sample))
                if postprocess:
                    sample = postprocess(sample)
                pipes['audio'].send((iteration - 1, rank, sample))
            model.train()

        epoch += 1

    for pipe in pipes.values():
        pipe.close()


def save_model(model, epoch, iteration, config, it=False):
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    checkpoint = model.save_networks()
    checkpoint.update({
        'iteration': iteration,
        'epoch': epoch,
        'timestamp': timestamp,
        'config': config
    })

    if it:
        name = f'iter_{iteration}'
    else:
        name = f'epoch_{epoch}'

    checkpoint_dir = config['checkpoint_dir']
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'{name}.pth'))

