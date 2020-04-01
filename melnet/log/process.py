from .utils import spec_to_image, spec_to_audio
from .logger import Logger


def logging_process(run_dir, proc_num, event, pipes, **audio_kwargs):
    if proc_num != 0:
        return

    print(event)
    print(event.is_set())
    event.wait()
    print('AWAKE')
    logger = Logger(run_dir)
    pipes = [(r, n, p) for r, v in pipes.items() for n, p in v.items()]
    print(pipes)

    def add_loss(name, iteration, losses):
        print(name, iteration, losses)
        for i, loss in enumerate(losses):
            logger.add_scalar(f'name/{i}', loss, iteration)

    def add_spectrogram(iteration, rank, spec):
        def image_callback(res):
            logger.add_image(f'spectrogram/{rank}', res, iteration)

        def audio_callback(res):
            logger.add_audio(f'audio/{rank}', res, iteration,
                             sr=audio_kwargs['sample_rate'])

        if len(spec.size()) > 2:
            spec = spec[0, :, :]
        spec = spec.cpu().transpose(0, 1).numpy()
        logger.add_async(spec_to_image, image_callback,
                         spec, **audio_kwargs)
        logger.add_async(spec_to_audio, audio_callback,
                         spec, **audio_kwargs)

    while event.is_set():
        for rank, name, pipe in pipes:
            print(pipe.poll())
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
