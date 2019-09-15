import torch
import time
import os
import math

from datetime import datetime
from torchvision.transforms import Compose
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from data import get_dataset, get_dataloader
from model import MelNetModel
from audio import MelScale, Spectrogram, PowerToDB
from utils import get_grad_plot


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

        self.model = MelNetModel(config)
        self.model.load_networks(checkpoint)
        self.config = config

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
            PowerToDB(normalized=True)
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

    def train(self):
        # Prepare Model
        self.model.train()
        # Get Dataloader
        dataloader = self.get_data('train')

        t = None
        start = self.epoch

        writer = SummaryWriter(self.config.run_dir, purge_step=self.iteration)
        writing_executor = ThreadPoolExecutor(max_workers=4)
        futures = dict()

        try:
            for epoch in range(start, self.config.epochs + 1):
                self.epoch = epoch

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
                        writer.add_scalar(f'loss/{i+1}', l, self.iteration)

                    # Gradient Flow
                    if self.config.log_grad is True and (self.iteration - 1) % 50 == 0:
                        for i, grad_info in enumerate(grad_infos):
                            future = writing_executor.submit(get_grad_plot, grad_info)
                            futures[future] = (self.iteration, i)

                        done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
                        for future in done:
                            it, net = futures[future]
                            try:
                                image = future.result()
                            except TimeoutError:
                                print('TimeoutError, no need to be too upset')
                            except Exception as exc:
                                print(f'Image for network {net} at iteration {it} returned exception: {exc}')
                            else:
                                del futures[future]
                                writer.add_image(f'gradient/{net+1}', image, it)

                    if any(math.isnan(x) for x in losses):
                        raise NaNError('NaN!!!')

                    if self.iteration % self.config.iter_interval == 0:
                        print(f"Storing network for iteration {self.iteration}")
                        self.save_model(True)

                    if epoch == start and batch == 0:
                        print('Training Started')

                    self.iteration += 1
                # TODO: Run test
                # self.test(mode='validation')
                # Store Network on epoch intervals
                if epoch % self.config.epoch_interval == 0:
                    print(f"Storing network for epoch {epoch}")
                    self.save_model()
        except NaNError:
            print("NaN!")
        finally:
            writer.close()
            writing_executor.shutdown(wait=True)

    def test(self, mode='test'):
        self.model.eval()

        size = self.config.dataset_size // 10 if self.config.mode != mode else None
        dataloader = self.get_data(mode, size=size)

        test_losses = list()
        for batch, x in enumerate(dataloader):
            losses = self.model.step(x)
            test_losses.append(losses)

        # TODO: process test_losses

    def sample(self):
        sample = self.model.sample()
        # TODO store the sample
        # generate spectrogram
        # Postprocess
        return sample


class NaNError(Exception):
    pass
