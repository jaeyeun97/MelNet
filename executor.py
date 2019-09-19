import torch
import time
import os
import math
import numpy as np

from datetime import datetime
from torchvision.transforms import Compose

from data import get_dataset, get_dataloader
from model import MelNetModel
from utils import get_grad_plot, get_spectrogram
from logger import Logger
from audio import (MelScale, Spectrogram, PowerToDB, Normalize,
                   MelToLinear, InverseSpectrogram, DBToPower, Denormalize)


class Executor(object):
    def __init__(self, config):
        os.makedirs(config.run_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.load_model(config)
        self.setup_preprocess()

    def load_model(self, config):
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
            config.load_config(checkpoint['config'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            time = str(datetime.fromtimestamp(checkpoint['timestamp']))
            print(f"Loading Epoch {self.epoch} from {time}")
        else:
            checkpoint = None
            self.epoch = 1
            self.iteration = (self.epoch - 1) * config.dataset_size + 1
        self.config = config
        self.model = MelNetModel(config)
        self.model.load_networks(checkpoint)

    def save_model(self, it=False):
        now = datetime.now()
        timestamp = datetime.timestamp(now)

        if it:
            name = f'iter_{self.iteration}'
        else:
            name = f'epoch_{self.epoch}'

        data = {
                'config': self.config.get_config(),
                'epoch': self.epoch,
                'iteration': self.iteration,
                'timestamp': timestamp
            }
        checkpoint = self.model.save_networks()
        data.update(checkpoint)
        torch.save(data, os.path.join(self.config.checkpoint_dir, f'{name}.pth'))

    def setup_preprocess(self):

        self.preprocess = Compose([
            Spectrogram(n_fft=self.config.n_fft,
                        win_length=self.config.win_length,
                        hop_length=self.config.hop_length,
                        normalized=True),
            MelScale(sample_rate=self.config.sample_rate,
                     n_fft=self.config.n_fft,
                     n_mels=self.config.n_mels),
            PowerToDB(),
            Normalize(self.config.top_db)
        ])

        self.denormalize = Denormalize(self.config.top_db)

        self.postprocess = Compose([
            DBToPower(),
            MelToLinear(sample_rate=self.config.sample_rate,
                        n_fft=self.config.n_fft,
                        n_mels=self.config.n_mels),
            InverseSpectrogram(n_fft=self.config.n_fft,
                               win_length=self.config.win_length,
                               hop_length=self.config.hop_length,
                               normalized=True)
        ])

        if self.config.preprocess_device != 'cpu':
            self.model.set_preprocess(self.preprocess)

    def get_data(self, mode, size=None):
        "Allow for custom mode and dataset size"
        size = self.config.dataset_size if size is None else size

        if self.config.preprocess_device == 'cpu':
            dataset = get_dataset(self.config, mode, size, preprocess=self.preprocess)
        else:
            dataset = get_dataset(self.config, mode, size)

        print(f"Dataset Size: {len(dataset)}")
        dataloader = get_dataloader(self.config, dataset)
        return dataloader

    def logging_decorator(fn):
        def ret(self, **kwargs):
            start = False
            if self.config.logging and 'logger' not in kwargs:
                kwargs['logger'] = Logger(self.config.run_dir,
                                          purge_step=self.iteration)
                start = True

            try:
                fn(self, **kwargs)
            finally:
                if start:
                    kwargs['logger'].close()
        return ret

    @logging_decorator
    def train(self, logger=None):
        # Prepare Model
        self.model.train()
        # Get Dataloader
        dataloader = self.get_data('train')

        t = None
        while self.epoch < self.config.epochs + 1:
            # Minibatch
            for batch, x in enumerate(dataloader):
                # Timer
                if batch % self.config.time_interval == 0:
                    new_t = time.time()
                    if t is not None:
                        print(f"Time executed: {new_t - t}")
                    t = new_t

                losses, grad_infos = self.model.step(x)

                # Logging Loss
                for i, l in enumerate(losses):
                    logger.add_scalar(f'train_loss/{i+1}', l, self.iteration)

                # Logging Gradient Flow
                if self.config.log_grad and (self.iteration - 1) % 50 == 0:
                    for i, grad_info in enumerate(grad_infos):
                        logger.add_async_image('gradient/{i+1}', get_grad_plot, grad_info, self.iteration)

                # NaN Check
                if any(math.isnan(x) for x in losses):
                    raise NaNError('NaN!!!')

                if self.iteration % self.config.iter_interval == 0:
                    print(f"Storing network for iteration {self.iteration}")
                    self.save_model(True)
                self.iteration += 1

            # Validation
            losses = self.evaluate(mode='validation')
            # Logging Eval Loss
            for i, l in enumerate(losses):
                logger.add_scalar(f'val_loss/{i+1}', l, self.iteration)

            if self.epoch % self.config.sample_interval == 0:
                # Sample
                sample = self.sample(logger=logger)
                # TODO To audio

            # Store Network on epoch intervals
            if self.epoch % self.config.epoch_interval == 0:
                print(f"Storing network for epoch {self.epoch}")
                self.save_model()
            self.model.train()
            self.epoch += 1

    def evaluate(self, mode='test'):
        self.model.eval()
        size = self.config.dataset_size
        size = max(1, min(100, size // 10)) if self.config.mode != mode else size
        dataloader = self.get_data(mode, size=size)

        with torch.no_grad():
            losses = [self.model.step(x, mode)[0] for x in dataloader]
            losses = np.array(losses)

        return np.mean(losses, axis=0)

    @logging_decorator
    def sample(self, logger=None):
        self.model.eval()
        with torch.no_grad():
            sample = self.model.sample()
        sample = self.denormalize(sample)
        if logger is not None:
            logger.add_async_image('spectrogram', get_spectrogram, sample, self.iteration)
        audio = self.postprocess(sample)
        if logger is not None:
            logger.add_audio('audio', audio, self.iteration)


class NaNError(Exception):
    pass
