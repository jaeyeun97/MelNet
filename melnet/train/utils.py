import os
import torch

from apex import amp
from datetime import datetime

def save_model(config, model, epoch, iteration, it=False):
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'timestamp': datetime.timestamp(datetime.now()),
        'model': model.state_dict(),
        'optimizer': model.optimizer_state_dict(),
        'amp': amp.state_dict()
    }

    name =  f'iter_{iteration}' if it else f'epoch_{epoch}'
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, f'net_{name}.pth'))
    torch.save(config, os.path.join(config.checkpoint_dir, f'config_{name}.pth'))


def load_model(config, model):
    if not config.resume:
        return 1, 1
    if config.load_iter != 0:
        name = f"net_iter_{config.load_iter}"
    elif config.load_epoch != 0:
        name = f"net_epoch_{config.load_epoch}"
    else:
        raise ValueError('Specify an iter or epoch to load from')

    if config.load_from is None:
        path = config.checkpoint_dir
    else:
        path = os.path.join(os.path.dirname(config.checkpoint_dir.rstrip('/')), config.load_from)

    path = os.path.join(path, f'{name}.pth')
    checkpoint = torch.load(path, map_location=torch.device('cuda', torch.cuda.current_device()))
    time = str(datetime.fromtimestamp(checkpoint['timestamp']))
    print(f"Loading Epoch {checkpoint['epoch']} from {time}")
    model.load_state_dict(checkpoint['model'])
    model.optimizer_load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    return checkpoint['epoch'] + 1, checkpoint['iteration'] + 1


def take(n, iterator):
    for i in range(n):
        try:
            yield next(iterator)
        except StopIteration:
            return
