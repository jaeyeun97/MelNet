from .utils import spec_to_image, spec_to_audio
from .logger import Logger


def logging_process(proc_num, config, event, pipes):
    if proc_num != 0:
        return

    logger = Logger(config.run_dir)
    pipes = [(r, n, p) for r, v in pipes.items() for n, p in v.items()]

    def add_loss(name, iteration, losses):
        for i, loss in enumerate(losses):
            logger.add_scalar(f'name/{i}', loss, iteration)

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
            if pipe.poll():
                content = pipe.recv()
                if name == 'train_loss':
                    add_loss('loss/train', *content)
                elif name == 'val_loss':
                    add_loss('loss/val', *content)
                elif name == 'test_loss':
                    add_loss('loss/test', *content)
                elif name == 'spectrogram':
                    add_spectrogram(*content)
        logger.process_async()

    print("Logger Exit")
    event.clear()
