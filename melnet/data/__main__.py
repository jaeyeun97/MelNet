import numpy as np
from absl import app, flags
from .dataset import Dataset
from .dataloader import DataLoader


dataset = Dataset('maestro', '/home/jaeyeun/KAIST/maestro-v2.0.0', split='train',
                  batch_size=1, sample_rate=22050,
                  num_workers=6, center=False,
                  timesteps=256, world_size=2)
# dataset.len = 10
dataloader = DataLoader(dataset, 6, 1)

def print_dataloader(dl):
    for data in dl:
        print(data[0], data[1].size(), data[2])
