import traceback

from functools import partial

from .process import logging_process

def get_log_proc_fn(config):
    return partial(logging_process, config.run_dir,
                   spec_type=config.spectrogram,
                   hop_length=config.hop_length,
                   sample_rate=config.sample_rate,
                   n_fft=config.n_fft,
                   n_mels=config.n_mels,
                   center=config.center,
                   bins_per_octave=config.bins_per_octave)

def debug_wrapper(*args, **kwargs):
    try:
        logging_process(*args, **kwargs)
    except Exception as e:
        print('---Caught exception in logging process---')
        traceback.print_exc()
        raise e
