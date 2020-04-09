import torch
import numpy as np

from .utils import spec_to_image, spec_to_audio
from .logger import Logger


def logging_process(proc_num, config, event, pipes):
    if proc_num != 0:
        return

    world_size = len(config.devices)
    logger = Logger(config.run_dir)
    pipes = [(r, n, p) for r, v in pipes.items() for n, p in v.items()]
    train_store = {}
    val_store = {}
    test_store = {}

    def add_loss(store, name, iteration, rank, losses):
        if isinstance(losses, torch.Tensor):
            losses = losses.numpy()
        if iteration not in store:
            store[iteration] = []
        store[iteration].append(losses)
        if len(store[iteration]) == world_size:
            losses = np.stack(store[iteration], axis=0).mean(axis=0)
            for i, loss in enumerate(losses):
                logger.add_scalar(f'loss_{i}/{name}', loss, iteration)
            del store[iteration]

    def add_spectrogram(iteration, rank, spec):
        def image_callback(res):
            logger.add_image(f'spectrogram/{rank}', res, iteration)

        def audio_callback(res):
            logger.add_audio(f'audio/{rank}', res, iteration,
                             sr=config.sample_rate)

        if len(spec.size()) > 2:
            spec = spec[0, :, :]
        spec = spec.cpu().transpose(0, 1).numpy()
        logger.add_async(spec_to_image, image_callback,
                         spec, config)
        logger.add_async(spec_to_audio, audio_callback,
                         spec, config)

    event.wait()
    while event.is_set():
        for rank, name, pipe in pipes:
            if not pipe.closed and pipe.poll():
                content = pipe.recv()
                if name == 'train_loss':
                    add_loss(train_store, 'train', *content)
                elif name == 'val_loss':
                    add_loss(val_store, 'val', *content)
                elif name == 'test_loss':
                    add_loss(test_store, 'test', *content)
                elif name == 'spectrogram':
                    add_spectrogram(*content)
        logger.process_async()

    print("Logger Exit")
