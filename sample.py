import os
import torch
import soundfile as sf

from functools import partial
from datetime import datetime

"""
Sampling Functions that can be applied or spawned as a separate thread.
"""


def get_sampling_fn(config, model_fn, path, postprocess,
                    pipes=None, iteration=0, rank=-1):
    return partial(sampling_process, model_fn, config['checkpoint_dir'], path,
                   postprocess=postprocess, load_epoch=config['load_epoch'],
                   load_iter=config['load_iter'],
                   pipes=pipes, iteration=iteration, rank=-1)


def sampling_process(model_fn, checkpoint_dir, path, device,
                     postprocess=None, load_epoch=0, load_iter=0,
                     pipes=None, iteration=0, rank=-1,
                     sample_rate=22050):

    model = model_fn(device)
    checkpoint = load_model(device, checkpoint_dir,
                            load_epoch=load_epoch,
                            load_iter=load_iter)
    model.load_networks(checkpoint)
    # path = os.path.join(sample_dir, f'{iteration}/{rank}.wav')
    sample(model, postprocess, path, sample_rate=sample_rate,
           pipes=pipes, iteration=iteration, rank=rank)


def sample(model, postprocess, save_path, sample_rate=22050,
           pipes=None, iteration=0, rank=-1):
    if pipes is not None and iteration > 0 and rank >= 0:
        logging = True
    else:
        logging = False

    with torch.no_grad():
        model.eval()
        sample = model.sample()
    if logging:
        pipes['spectrogram'].send((iteration, rank, sample.cpu()))
    audio = postprocess(sample)
    if len(audio.size()) > 1:
        audio = audio.squeeze(0)
    audio = audio.cpu().numpy()
    if logging:
        pipes['audio'].send((iteration, rank, audio))
    if save_path is not None:
        sf.write(save_path, audio, sample_rate)


def load_model(device, checkpoint_dir, load_epoch=0, load_iter=0):
    if load_iter != 0:
        name = f"net_iter_{load_iter}"
    elif load_epoch != 0:
        name = f"net_epoch_{load_epoch}"
    else:
        raise RuntimeError('Do not have a model to load')

    if name is not None:
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{name}.pth'),
                                map_location=device)
        time = str(datetime.fromtimestamp(checkpoint['timestamp']))
        print(f"Loading Epoch {checkpoint['epoch']} from {time}")
    else:
        checkpoint = None
    return checkpoint

