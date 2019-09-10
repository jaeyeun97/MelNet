from torch.utils.data import DataLoader

from maestro import Maestro


def get_dataloader(config):
    if config.dataset == 'maestro':
        dataset = Maestro(config.dataroot, config.frame_length,
                          epoch_size=config.epoch_size,
                          sample_rate=config.sample_rate,
                          split=config.mode)
    else:
        raise NotImplementedError()

    print(f"Dataset size: {len(dataset)}")
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

