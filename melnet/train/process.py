import os
import time
import torch
import torch.distributed as dist

from ..model import MelNet
from ..data import DataLoader
from .utils import load_model, save_model, take

def train(rank, world_size, config, pipes, train_dataset, val_dataset, seed):
    torch.cuda.set_device(config.devices[rank])

    if config.logging:
        pipes = pipes[rank]

    distributed = world_size > 1

    if seed is not None:
        torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)


    n_freq = config.n_mels if config.spectrogram == 'mel' else config.n_bins
    model = MelNet(config.width, n_freq, config.n_layers, config.n_mixtures, config.mode,
                   optimizer_cls=config.optimizer, optim_args=config.optim_args,
                   grad_acc=config.grad_acc, grad_clip=config.grad_clip,
                   amp_level=config.amp_level, profile=config.prof,
                   distributed=distributed)

    epoch, iteration = load_model(config, model)
    val_i = iteration


    train_dataset.set_rank(rank)
    val_dataset.set_rank(rank)
    train_loader = DataLoader(train_dataset, config.num_workers, config.batch_size)
    val_loader = DataLoader(val_dataset, config.num_workers, config.batch_size)

    t = time.time()
    while epoch <= config.epochs:
        model.zero_grad()
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        while True:
            if config.prof:
                torch.cuda.nvtx.range_push(f'Iteration {iteration}')
                torch.cuda.nvtx.range_push('Data Fetch')
            try:
                entries, x, flag_lasts = next(train_iter)
            except StopIteration:
                break

            if config.prof: torch.cuda.nvtx.range_pop()

            step_iter = iteration % config.grad_acc == 0

            if config.prof: torch.cuda.nvtx.range_push('Step')

            losses = model(x, entries, flag_lasts,
                           step_iter=step_iter)

            if config.prof: torch.cuda.nvtx.range_pop()

            # Logging Loss
            if config.logging and iteration % config.log_interval == 0:
                losses = torch.stack(losses)
                pipes['train_loss'].send((iteration, rank, losses))
            del losses

            # Validate
            if config.logging and iteration % config.val_interval == 0:
                val_i = validate(config, model, val_iter, pipes['val_loss'],
                                 rank, world_size, val_i)

            # Save
            if iteration % config.iter_interval == 0:
                if rank == 0:
                    print(f"Storing network for iteration {iteration}")
                    save_model(config, model, epoch, iteration, True)
                if distributed: dist.barrier()

            # Time
            if iteration % config.time_interval == 0 and rank == 0:
                t, old_t = time.time(), t
                print(f'Time executed: {t - old_t}')

            if config.prof: torch.cuda.nvtx.range_pop()

            if config.prof and iteration == config.prof_end:
                print(f'Ending profiling on Iteration {iteration}')
                break
            iteration += 1

        del train_iter
        del val_iter

        if config.prof and iteration == config.prof_end:
            break

        model.step()

        # Store Network on epoch intervals
        if epoch % config.epoch_interval == 0:
            if rank == 0:
                print(f"Storing network for epoch {epoch}")
                save_model(config, model, epoch, iteration)
            if distributed: dist.barrier()

        # Sample
        if config.logging and epoch % config.sample_interval == 0:
            with torch.no_grad():
                sample = model.sample(config.timesteps)
            sample = (sample - 1) * 80  # Undo normalization
            # postprocessing on the logger thread
            pipes['spectrogram'].send((iteration - 1, rank, sample))
            model.train()

        epoch += 1

    print('Closing Pipes')
    if pipes:
        for pipe in pipes.values():
            pipe.close()

    # if distributed:
    #     dist.destroy_process_group()



def validate(config, model, val_iter, pipe, rank, world_size, val_i):
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(take(config.val_interval // 10, val_iter)):
            entries, x, flag_lasts = x
            losses = model(x, entries, flag_lasts)

            if i % (config.log_interval // 10) == 0:
                pipe.send((val_i, rank, losses))
            val_i += 10
    model.train()
    return val_i
