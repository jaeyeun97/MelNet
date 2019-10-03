import torch
import time
import os
import math
import numpy as np

from datetime import datetime
from torchvision.transforms import Compose, Lambda

from data import get_dataset, get_dataloader
from model import MelNetModel
from utils import get_grad_plot, get_spectrogram
from logger import Logger
from audio import (MelScale, Spectrogram, LogAmplitude,
                   MelToLinear, InverseSpectrogram, LinearAmplitude)


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
            self.epoch = checkpoint['epoch'], Lambda
            self.iteration = checkpoint['iteration']
            time = str(datetime.fromtimestamp(checkpoint['timestamp']))
            print(f"Loading Epoch {self.epoch} from {time}")
        else:
            checkpoint = None
            self.epoch = 0
            self.iteration = 0
        self.iteration += 1
        self.epoch += 1
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
            Lambda(lambda x: x.to(self.config.preprocess_device)),
            Spectrogram(n_fft=self.config.n_fft,
                        win_length=self.config.win_length,
                        hop_length=self.config.hop_length,
                        power=1),
            MelScale(sample_rate=self.config.sample_rate,
                     n_fft=self.config.n_fft,
                     n_mels=self.config.n_mels),
            LogAmplitude()
        ])

        self.postprocess = Compose([
            LinearAmplitude(),
            MelToLinear(sample_rate=self.config.sample_rate,
                        n_fft=self.config.n_fft,
                        n_mels=self.config.n_mels),
            InverseSpectrogram(n_fft=self.config.n_fft,
                               win_length=self.config.win_length,
                               hop_length=self.config.hop_length,
                               length=self.config.frame_length,
                               power=1)
        ])

    def get_data(self, mode, size=None):
        "Allow for custom mode and dataset size"
        size = self.config.dataset_size if size is None else size
        dataset = get_dataset(self.config, mode, size, preprocess=self.preprocess)

        print(f"Dataset Size: {len(dataset)}")
        dataloader = get_dataloader(self.config, dataset)
        return dataloader

    def logging_decorator(fn):
        def ret(self, **kwargs):
            start = False
            if self.config.logging and 'logger' not in kwargs:
                kwargs['logger'] = Logger(self.config.run_dir)
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
        train_dataloader = self.get_data('train')
        val_dataloader = self.get_data('validation', size=10*self.config.batch_size)

        t = None
        while self.epoch < self.config.epochs + 1:
            # Minibatch
            for batch, x in enumerate(train_dataloader):
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
                        logger.add_async_image('gradient/{i+1}', get_grad_plot, self.iteration, grad_info)
                    logger.process_async()

                # NaN Check
                if any(math.isnan(x) for x in losses):
                    raise NaNError('NaN!!!')

                # Save
                if self.iteration % self.config.iter_interval == 0:
                    print(f"Storing network for iteration {self.iteration}")
                    self.save_model(True)

                # Validate
                if (self.iteration - 1) % self.config.val_interval == 0:
                    with torch.no_grad():
                        self.model.eval()
                        losses = [self.model.step(x, 'validation')[0] for x in val_dataloader]
                        losses = np.mean(losses, axis=0)

                        for i, l in enumerate(losses):
                            logger.add_scalar(f'val_loss/{i+1}', l, self.iteration)
                    self.model.train()
                self.iteration += 1

            # Store Network on epoch intervals
            if self.epoch % self.config.epoch_interval == 0:
                print(f"Storing network for epoch {self.epoch}")
                self.save_model()

            if self.epoch % self.config.sample_interval == 0:
                self.sample(logger=logger)
                logger.process_async()
                self.model.train()

            self.epoch += 1

    def test(self):
        self.model.eval()
        dataloader = self.get_data('test')

        with torch.no_grad():
            losses = [self.model.step(x, 'test')[0] for x in dataloader]

        return np.mean(losses, axis=0)

    @logging_decorator
    def sample(self, logger=None):
        self.model.eval()
        with torch.no_grad():
            # 1, T, M
            sample = self.model.sample()
        if logger is not None:
            for i in range(sample.size(0)):
                logger.add_async_image(f'spectrogram/{i}', get_spectrogram,
                                       self.iteration, sample[i, :, :].mul(20),
                                       hop_length=self.config.hop_length,
                                       sr=self.config.sample_rate)
        audio = self.postprocess(sample)
        if logger is not None:
            for i in range(audio.size(0)):
                logger.add_audio(f'audio/{i}', audio[i, :], self.iteration)


class NaNError(Exception):
    pass
