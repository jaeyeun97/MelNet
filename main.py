import os
import torch
import torch.multiprocessing as mp

from datetime import datetime

from config import Config
from train import get_train_fn
from model import get_model_fn
from data import get_dataset
from tb import get_log_proc_fn
from audio import get_audio_processes


def load_model(config):
    if config.load_iter != 0:
        name = f'iter_{config.load_iter}'
        flag = True
    elif config.load_epoch != 0:
        name = f'epoch_{config.load_epoch}'
        flag = True
    elif config.mode != 'train':
        raise Exception('Load Network for Validation, Testing, or Sampling')
    else:
        flag = False

    if flag:
        checkpoint = torch.load(os.path.join(config.checkpoint_dir, f'{name}.pth'))
        time = str(datetime.fromtimestamp(checkpoint['timestamp']))
        print(f"Loading Epoch {checkpoint['epoch']} from {time}")
    else:
        checkpoint = None
    return checkpoint


def get_pipe_types(mode):
    if mode == 'train':
        return ['train_loss', 'val_loss', 'model', 'spectrogram', 'audio']
    elif mode == 'sample':
        return ['spectrogram', 'audio']


def main():
    config = Config()  

    # Load Model
    checkpoint = load_model(config)
    if checkpoint is not None:
        config = config.load_config(checkpoint['config'])

    # Process #
    world_size = len(config.devices)
    # Audio Processes
    preprocess, postprocess = get_audio_processes(config)
    # pre-apply everything except device
    model_fn = get_model_fn(config)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.makedirs(config.run_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Logging Thread Spawn
    if config.logging:
        log_p = dict()
        worker_p = dict()
        for rank in range(world_size):
            worker_p[rank] = dict()
            log_p[rank] = dict()

            # Normal Operations
            for pt in get_pipe_types(config.mode):
                log_p[rank][pt], worker_p[rank][pt] = mp.Pipe(False)

            # Gradient Logging
            if config.log_grad:
                log_p[rank]['grad'], worker_p[rank]['grad'] = mp.Pipe()
        ctrl_r, ctrl_w = mp.Pipe(False)
        log_proc_fn = get_log_proc_fn(config)
        logging_ctx = mp.spawn(log_proc_fn, args=(world_size, ctrl_r, log_p),
                               join=False, nprocs=1)
    else:
        log_p = None
        worker_p = None
        logging_ctx = None

    # Training Threads Spawn
    if config.mode == 'train':
        train_dataset = get_dataset(config, 'train',
                                    preprocess=preprocess)
        val_dataset = get_dataset(config, 'validation',
                                  size=10*world_size*config.batch_size,
                                  preprocess=preprocess)

        # pre-apply everything except rank, world_size, device, pipes
        train_fn = get_train_fn(config, model_fn, train_dataset, val_dataset, checkpoint, postprocess)
        mp.spawn(train_fn, args=(world_size, config.devices, worker_p), nprocs=world_size)

    # TODO: low priority
    elif config.mode == 'test':
        test_dataset = get_dataset(config, 'test', preprocess=preprocess)
    # TODO: low priority
    elif config.mode == 'sample':
        pass

    if logging_ctx is not None:
        ctrl_w.send(True)
        logging_ctx.join()


if __name__ == "__main__":
    main()
