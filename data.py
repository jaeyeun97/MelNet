from maestro import Maestro


def get_dataset(config, mode, size=None, preprocess=None):
    size = config.dataset_size if size is None else size
    if config.dataset == 'maestro':
        dataset = Maestro(config.dataroot, config.frame_length,
                          size=size, sample_rate=config.sample_rate,
                          split=mode, preprocess=preprocess)
    else:
        raise NotImplementedError()

    return dataset
