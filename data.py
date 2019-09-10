from torch.utils.data import DataLoader

from maestro import Maestro


def get_dataset(config, mode, size, preprocess=None):
    if config.dataset == 'maestro':
        dataset = Maestro(config.dataroot, config.frame_length,
                          size=size, sample_rate=config.sample_rate,
                          split=mode, preprocess=preprocess)
    else:
        raise NotImplementedError()

    return dataset


def get_dataloader(config, dataset):
    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        shuffle=config.shuffle,
                        pin_memory=True)

    return loader


if __name__ == "__main__":
    from config import Config
    config = Config()
    data_loader = get_dataloader(config)
    __import__('ipdb').set_trace()

