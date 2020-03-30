import time
import torch
import torch.distributed as dist

from ..model import MelNet
from ..data import DataLoader
from .utils import load_model, save_model, take

def train(rank, world_size, config, pipes, train_dataset, val_dataset, seed):
    torch.cuda.set_device(config.devices[rank])
    pipes = pipes[rank]

    if seed is not None:
        torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    n_freq = config.n_mels if config.spectrogram == 'mel' else config.n_bins
    model = MelNet(config.width, n_freq, config.n_layers, config.n_mixtures, config.mode,
                   optimizer_cls=config.optimizer, optim_args=config.optim_args,
                   grad_acc=config.grad_acc, amp_level=config.amp_level, distributed=True)

    epoch, iteration = load_model(config, model)


    train_dataset.set_rank(rank)
    val_dataset.set_rank(rank)
    train_loader = DataLoader(train_dataset, config.num_workers, config.batch_size)
    val_loader = DataLoader(val_dataset, config.num_workers, config.batch_size)

    t = time.time()
    while epoch < config.epochs:
        model.zero_grad()
        val_iter = iter(val_loader)
        for batch, (entries, x, flag_lasts) in enumerate(train_loader):
            losses = torch.rand(len(config.n_layers)) # model(x, entries, flag_lasts)
            print(iteration, entries, x.size(), flag_lasts)
            print(losses)

            if iteration % config.grad_acc == 0:
                model.step()
                model.zero_grad()

            if torch.isnan(losses).any():
                print("NaN at loss")

            # Logging Loss
            # TODO: do this on the logger process
            if iteration % config.log_interval == 0:
                dist.reduce(losses, 0)
                if rank == 0:
                    losses = losses.div(world_size).cpu().numpy()
                    pipes['train_loss'].send((iteration, losses))

            # Validate
            if iteration % config.val_interval == 0:
                validate(config, model, val_loader, pipes['val_loss'],
                         rank, world_size, iteration)

            # Save
            if iteration % config.iter_interval == 0 and rank == 0:
                print(f"Storing network for iteration {iteration}")
                save_model(config, model, epoch, iteration, True)

            # Time
            if iteration % config.time_interval == 0 and rank == 0:
                t, old_t = time.time(), t
                print(f'Time executed: {t - old_t}')
            iteration += 1

        model.step()
        del val_iter

        # Store Network on epoch intervals
        if epoch % config.epoch_interval == 0 and rank == 0:
            print(f"Storing network for epoch {epoch}")
            save_model(config, model, epoch, iteration)

        # Sample
        if epoch % config.sample_interval == 0:
            with torch.no_grad():
                sample = model.sample(config.timesteps)
            sample = (sample - 1) * 80  # Undo normalization
            # postprocessing on the logger thread
            pipes['spectrogram'].send((iteration - 1, rank, sample))
            model.train()

        epoch += 1

    for pipe in pipes.values():
        pipe.close()

    dist.destroy_process_group()



def validate(config, model, val_iter, pipe, rank, world_size, iteration):
    with torch.no_grad():
        for i in take(config.val_interval // 10, val_iter):
            entries, x, flag_lasts = next(val_iter)
            losses = model(x, entries, flag_lasts)

            if i % config.log_interval:
                dist.reduce(losses, 0)
                if rank == 0:
                    losses = losses.div(world_size).cpu().numpy()
                    pipe.send((iteration, losses))
