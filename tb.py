import traceback

from functools import partial
from tensorboardX import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
# from pudb.remote import set_trace

from utils import get_spectrogram, get_grad_plot


def get_log_proc_fn(config):
    return partial(debug_wrapper, config.run_dir,
                   hop_length=config.hop_length,
                   sample_rate=config.sample_rate)


def debug_wrapper(*args, **kwargs):
    try:
        logging_process(*args, **kwargs)
    except Exception as e:
        print('---Caught exception in logging process---')
        traceback.print_exc()
        print()
        raise e

def logging_process(run_dir, proc_num, world_size, ctrl, pipes,
                    hop_length=256, sample_rate=22050):
    if proc_num != 0:
        return

    logger = Logger(run_dir)

    while not ctrl.poll():
        for rank, v in pipes.items():
            for name, pipe in v.items():
                if pipe.poll():
                    content = pipe.recv()
                    if name == 'train_loss':
                        iteration, losses = content
                        for i, loss in enumerate(losses):
                            logger.add_scalar(f'loss/train/{i}', loss, iteration)
                    elif name == 'val_loss':
                        iteration, losses = content
                        for i, loss in enumerate(losses):
                            logger.add_scalar(f'loss/val/{i}', loss, iteration) 
                    elif name == 'spectrogram':
                        iteration, rank, sample = content
                        logger.add_async_image(f'spectrogram/{rank}',
                                               get_spectrogram,
                                               iteration, sample.mul(20),
                                               hop_length=hop_length,
                                               sr=sample_rate)
                    elif name == 'audio':
                        iteration, rank, sample = content
                        logger.add_audio(f'audio/{rank}', sample, iteration,
                                         sr=sample_rate)
                    elif name == 'grad':
                        iteration, grad_infos = pipe.recv()
                        for i, grad_info in enumerate(grad_infos):
                            logger.add_async_image(f'gradient/{i+1}', get_grad_plot,
                                                   iteration, grad_info)
        logger.process_async()
    return ctrl.recv()


class Logger(object):
    def __init__(self, run_dir, **kwargs):
        self.writer = SummaryWriter(run_dir, **kwargs)
        self.async_executor = ThreadPoolExecutor(max_workers=4)
        self.futures = dict()

    def add_scalar(self, name, scalar, global_step):
        self.writer.add_scalar(name, scalar, global_step)

    def add_audio(self, name, audio, global_step, sr=22050):
        self.writer.add_audio(name, audio, global_step, sample_rate=sr)

    def add_async_image(self, name, fn, global_step, *args, **kwargs):
        future = self.async_executor.submit(fn, *args, **kwargs)
        self.futures[future] = (global_step, name)

    def process_async(self):
        done = list(filter(lambda future: future.done(), self.futures))

        for future in done:
            step, name = self.futures[future]
            try:
                image = future.result()
            except TimeoutError:
                print('TimeoutError, no need to be too upset')
            except Exception as exc:
                print(f'{name} at iteration {step} returned exception: {exc}')
            else:
                del self.futures[future]
                self.writer.add_image(name, image, step)

    def close(self):
        self.async_executor.shutdown(wait=True)
        self.process_async()
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

