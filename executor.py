import torch
import time

from torchvision.transforms import Compose

from data import get_dataset, get_dataloader
from model import MelNetModel
from audio import MelScale, Spectrogram


class Executor(object):
    def __init__(self, config):
        self.config = config
        self.model = MelNetModel(config)

        if self.config.load_iter != 0:
            self.model.load_networks(it=self.config.load_iter)
        elif self.config.load_epoch != 0:
            self.model.load_networks(epoch=self.config.load_epoch)
        elif self.config.load_timestamp != 0:
            self.model.load_networks(timestamp=self.config.load_timestamp)
        elif self.config.mode != 'train':
            raise Exception('Load Network for Validation, Testing, or Sampling')

        # Setup preprocess
        if self.config.preprocess_device == 'cpu':
            device = torch.device('cpu')
            dtype = None
        else:
            device = self.config.device
            dtype = self.config.dtype

        self.preprocess = Compose([
            Spectrogram(n_fft=self.config.n_fft,
                        win_length=self.config.win_length,
                        hop_length=self.config.hop_length,
                        normalized=True,
                        dtype=dtype,
                        device=device),
            MelScale(sample_rate=self.config.sample_rate,
                     n_fft=self.config.n_fft,
                     n_mels=self.config.n_mels,
                     dtype=dtype,
                     device=device)
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

        i = 1 
        t = None
        for epoch in range(self.config.load_epoch + 1, self.config.epochs + 1):
            for batch, x in enumerate(dataloader):
                # Timer
                if batch % 5 == 0:
                    new_t = time.time()
                    if t is not None:
                        print(f"Time executed: {new_t - t}")
                    t = new_t

                losses = self.model.step(x)

                # TODO: log to file and only print progress
                print("Epoch %s\tBatch %s\t%s" % (epoch, batch, '\t'.join(str(l) for l in losses)))

                if i % self.config.iter_interval == 0:
                    print(f"Storing network for iter {i}")
                    self.model.save_networks(it=i)
                i += 1

            # Run test
            # self.test(mode='validation')
            # Store Network on epoch intervals
            if epoch % self.config.epoch_interval == 0:
                print(f"Storing network for epoch {epoch}")
                self.model.save_networks(epoch=epoch)

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
        return sample
