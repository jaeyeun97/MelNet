from .dataset import Dataset
from .dataloader import DataLoader


def get_dataset(config, mode, world_size):
    return Dataset(config.dataset, config.dataroot,
                   split=mode, batch_size=config.batch_size,
                   sample_rate=config.sample_rate, timesteps=config.timesteps,
                   method=config.spectrogram, hop_length=config.hop_length,
                   center=config.center, n_fft=config.n_fft, n_mels=config.n_mels,
                   n_bins=config.n_bins, bins_per_octave=config.bins_per_octave,
                   num_workers=config.num_workers, world_size=world_size)
