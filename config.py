import os
import argparse
import pprint
import torch

from adabound import AdaBound


def get_optimizer(optimizer_name='Adam'):
    """Get optimizer by name"""
    # optimizer_name = optimizer_name.capitalize()
    if optimizer_name == 'AdaBound':
        return AdaBound
    return getattr(torch.optim, optimizer_name)


def argparse_dict(arg):
    result = dict()
    for s in arg.split(','):
        pairs = s.split('=')
        result[pairs[0]] = eval(pairs[1])
    return result


class Config(object):
    def __init__(self, config=None):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('name', type=str)
        self.parser.add_argument('--run-dir', type=str, default='./runs')
        self.parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
        self.parser.add_argument('--sample-dir', type=str, default='./samples')
        self.parser.add_argument('--load-iter', type=int, default=0)
        self.parser.add_argument('--load-epoch', type=int, default=0)
        self.parser.add_argument('--mode', type=str, default='train', help='[train | validation | test | sample]')
        self.parser.add_argument('--devices', nargs='+', type=int, default=[0])
        self.parser.add_argument('--logging', action='store_false')
        self.parser.add_argument('--top-db', type=float, default=80.0)

        opt, _ = self.parser.parse_known_args()

        # --- Mode Specific --- #

        if opt.mode != 'sample':
            self.parser.add_argument('--dataset', type=str, default='maestro', help='Which dataset to use: [maestro | musicnet]')
            self.parser.add_argument('--dataroot', type=str, required=True, help='parent directory of datasets')
            self.parser.add_argument('--batch-size', type=int, default=1)
            self.parser.add_argument('--num-workers', type=int, default=4)
            self.parser.add_argument('--shuffle', action='store_true')
            self.parser.add_argument('--dataset-size', type=int, default=4000)
            self.parser.add_argument('--preprocess-device', type=int, default=-1)

        if opt.mode == 'train':
            self.parser.add_argument('--val-interval', type=int, default=100)
            self.parser.add_argument('--time-interval', type=int, default=20)
            self.parser.add_argument('--iter-interval', type=int, default=2000)
            self.parser.add_argument('--log-grad', action='store_true')
            self.parser.add_argument('--epoch-interval', type=int, default=1)
            self.parser.add_argument('--sample-interval', type=int, default=10)
            self.parser.add_argument('--epochs', type=int, default=200)

        if opt.load_iter == 0 and opt.load_epoch == 0 and opt.mode == 'train':
            self.new_config()
            new_config = True
        else:
            new_config = False

        args = self.parser.parse_args()
        self.config = vars(args)
        self.initvars(new_config)

    def new_config(self): 
        # parser.add_argument('--offload-dir', type=str, default=f'/tmp/{os.getpid()}/')
        self.parser.add_argument('--dtype', type=str, default='float')

        # --- Audio --- #
        self.parser.add_argument('--sample-rate', type=int, default=22050)
        # TODO: implement CQT measures
        # parser.add_argument('--spectrogram', type=str, default='mel_stft', help='[mel_stft | cqt]')

        opt, _ = self.parser.parse_known_args()

        # --- Mel-STFT specific --- #
        self.parser.add_argument('--n-fft', type=int, default=2048)
        self.parser.add_argument('--n-mels', type=int, default=256)
        self.parser.add_argument('--win-length', type=int, default=None)
        self.parser.add_argument('--hop-length', type=int, default=None)
        self.parser.add_argument('--timesteps', type=int, default=256)

        if opt.mode == 'train':
            self.parser.add_argument('--optimizer', type=str, default='SGD')
            self.parser.add_argument('--lr-decay', action='store_true')
            self.parser.add_argument('--optim-args', type=argparse_dict, default={})
            self.parser.add_argument('--grad-clip', type=float, default=0.)
            self.parser.add_argument('--grad-scale', type=float, default=0.)

        # --- Network --- #
        # Network width
        self.parser.add_argument('--n-layers', nargs='+', type=int, default=[16, 6, 5, 4])
        self.parser.add_argument('--width', type=int, default=256)
        # Number of mixtures
        self.parser.add_argument('--mixtures', type=int, default=10)

        self.parser.add_argument('--amp-enable', action='store_false')
        self.parser.add_argument('--amp-level', type=str, default='O1')

    def initvars(self, new_config):
        self.config['run_dir'] = os.path.join(self.config['run_dir'],
                                              self.config['name'])
        self.config['checkpoint_dir'] = os.path.join(self.config['checkpoint_dir'],
                                                     self.config['name'])
        if not new_config:
            return

        self.config['optimizer'] = get_optimizer(self.config['optimizer']) 
        self.config['dtype'] = getattr(torch, self.config['dtype'])

        if self.config['win_length'] is None:
            self.config['win_length'] = self.config['n_fft']

        if self.config['hop_length'] is None:
            self.config['hop_length'] = self.config['n_fft'] // 4

        self.config['frame_length'] = self.config['hop_length'] * self.config['timesteps'] - 1

    def load_config(self, config):
        d = self.config
        if isinstance(config, Config):
            self.config = config.config
        else:
            self.config = config

        for k in d.keys():
            default = self.parser.get_default(k)
            if k not in self.config or (d[k] != default and self.config[k] != d[k]):
                self.config[k] = d[k]

    def get_config(self):
        return self.config

    def __getitem__(self, name):
        return self.config[name]

    def __setitem__(self, name, value):
        self.config[name] = value

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        else:
            super().__getattr__()

    def __repr__(self):
        return pprint.pformat(self.config)


if __name__ == "__main__":
    config = Config()
    print(config)
