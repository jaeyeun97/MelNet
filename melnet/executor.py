import os
import torch
import numpy as np
import torch.multiprocessing as mp

from .config import get_config
from .data import get_dataset
from .log import get_log_proc_fn
from .train import train


class Executor(object):
    def __init__(self):
        self.config = get_config()
        print(self.config)
        self.devices = [torch.device('cuda', d_id) for d_id in self.config.devices]
        self.world_size = len(self.devices)

        self.prep_env()
        if self.config.logging:
            self.prep_pipes()
        self.prep_datasets()

    def prep_env(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.makedirs(self.config.run_dir, exist_ok=True)
        os.makedirs(self.config.sample_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    @property
    def pipe_types(self):
        if self.config.mode == 'train':
            return ['train_loss', 'val_loss', 'spectrogram']
        elif self.config.mode == 'test':
            return ['test_loss']
        elif self.config.mode == 'sample':
            return ['spectrogram']

    def prep_pipes(self):
        self.log_p = dict()
        self.worker_p = dict()
        for rank in range(self.world_size):
            self.worker_p[rank] = dict()
            self.log_p[rank] = dict()
            for pt in self.pipe_types:
                self.log_p[rank][pt], self.worker_p[rank][pt] = mp.Pipe(False)
        self.logging = mp.Event()

    def prep_datasets(self):
        # No dataset is needed for sampling (yet).
        if self.config.mode == 'train':
            self.train_dataset = get_dataset(self.config, 'train', self.world_size)
            self.val_dataset = get_dataset(self.config, 'validation', self.world_size)
        elif self.config.mode == 'test':
            self.test_dataset = get_dataset(self.config, 'test', self.world_size)

    def spawn_logger(self):
        self.logging.set()
        log_proc_fn = get_log_proc_fn(self.config)
        self.logging_ctx = mp.spawn(log_proc_fn, args=(self.logging, self.log_p),
                                    join=False, nprocs=1)

    def spawn_trainer(self):
        seed = np.random.randint(np.iinfo(np.int).max)
        mp.spawn(train, args=(self.world_size, self.config, self.worker_p,
                              self.train_dataset, self.val_dataset, seed), nprocs=self.world_size)

    def spawn_sampler(self):
        raise NotImplementedError

    def spawn_tester(self):
        raise NotImplementedError

    def run(self):
        if self.config.logging:
            self.spawn_logger()

        if self.config.mode == 'train':
            self.spawn_trainer()
        elif self.config.mode == 'sample':
            self.spawn_sampler()
        elif self.config.mode == 'test':
            self.spawn_tester()

        if self.config.logging:
            self.logging.clear()
            self.logging_ctx.join()