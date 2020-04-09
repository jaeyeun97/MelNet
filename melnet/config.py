import os
import argparse
import pprint
import torch

from apex.optimizers import FusedAdam


def argparse_optimizer(arg):
    """Get optimizer by name"""
    if arg:
        if arg == "FusedAdam":
            return FusedAdam
        return getattr(torch.optim, arg)


def argparse_dict(arg):
    result = dict()
    for s in arg.split(','):
        pairs = s.split('=')
        result[pairs[0]] = eval(pairs[1])
    return result


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('name', type=str)
    opt, _ = parser.parse_known_args()

    parser.add_argument('--run-dir', type=lambda path: os.path.join(path, opt.name), default='./runs')
    parser.add_argument('--checkpoint-dir', type=lambda path: os.path.join(path, opt.name), default='./checkpoints')
    parser.add_argument('--sample-dir', type=lambda path: os.path.join(path, opt.name), default='./samples')
    parser.add_argument('--load-from', type=str)
    parser.add_argument('--load-iter', type=int, default=0)
    parser.add_argument('--load-epoch', type=int, default=0)
    parser.add_argument('--no-logging', action='store_true')
    parser.add_argument('--devices', nargs='+', type=int, default=list(range(torch.cuda.device_count())))
    parser.add_argument('--mode', type=str, default='train', help='[train | test | sample]')
    opt, _ = parser.parse_known_args()

    # --- Mode Specific --- #

    if opt.mode != 'sample':
        # Train and Test
        parser.add_argument('--batch-size', type=int)
        parser.add_argument('--num-workers', type=int)

    if opt.mode == 'train':
        parser.add_argument('--grad-acc', type=int)
        parser.add_argument('--grad-clip', type=int)
        parser.add_argument('--log-interval', type=int)
        parser.add_argument('--val-interval', type=int)
        parser.add_argument('--time-interval', type=int)
        parser.add_argument('--iter-interval', type=int)
        parser.add_argument('--epoch-interval', type=int)
        parser.add_argument('--sample-interval', type=int)
        parser.add_argument('--prof', action='store_true')
        parser.add_argument('--prof-end', type=int)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--optimizer', type=argparse_optimizer)
        # parser.add_argument('--lr-decay', action='store_true')
        parser.add_argument('--optim-args', type=argparse_dict)


    if opt.load_iter == 0 and opt.load_epoch == 0:
        # New Configuration
        assert opt.mode == 'train', 'Load checkpoint for testing or sampling!'
        assert opt.load_from is None, 'Load from an epoch or iter'

        parser.set_defaults(batch_size=2,
                            num_workers=8,
                            grad_acc=8,
                            grad_clip=0,
                            log_interval=20,
                            val_interval=200,
                            time_interval=20,
                            iter_interval=2000,
                            epoch_interval=1,
                            sample_interval=10,
                            prof_end=20,
                            epochs=200,
                            optimizer=torch.optim.RMSprop,
                            optim_args={'lr': 0.0001, 'momentum': 0.9})

        parser.add_argument('--dataset', type=str, default='maestro', help='Which dataset to use: [maestro | musicnet]')
        parser.add_argument('--dataroot', type=str, required=True, help='parent directory of datasets')
        parser.add_argument('--sample-rate', type=int, default=22050)
        parser.add_argument('--spectrogram', type=str, default='mel', help='[mel | cqt]')
        parser.add_argument('--timesteps', type=int, default=256)
        parser.add_argument('--hop-length', type=int, default=256)
        parser.add_argument('--n-mixtures', type=int, default=10)
        # --- Mel-STFT specific --- #
        parser.add_argument('--n-fft', type=int, default=2048)
        parser.add_argument('--n-mels', type=int, default=256)
        parser.add_argument('--center', action='store_true')
        # --- CQT --- #
        parser.add_argument('--n-bins', type=int, default=84 * 3)
        parser.add_argument('--bins-per-octave', type=int, default=36)
        # --- Network --- #
        # Network width
        parser.add_argument('--n-layers', nargs='+', type=int, default=[16, 6, 5, 4])
        parser.add_argument('--width', type=int, default=512)
        # Number of mixtures
        parser.add_argument('--mixtures', type=int, default=10)
        parser.add_argument('--amp-level', type=str, default='O1')
    else:
        parser.add_argument('--dataroot', type=str, help='parent directory of datasets')

    args = parser.parse_args()
    config = vars(args)

    config['logging'] = not config['no_logging']
    del config['no_logging']

    if args.load_iter != 0:
        load_name = f"config_iter_{config['load_iter']}"
        resume = True
    elif args.load_epoch != 0:
        load_name = f"config_epoch_{config['load_epoch']}"
        resume = True
    else:
        resume = False

    if resume:
        if args.load_from is not None:
            load_path = os.path.join(os.path.dirname(args.checkpoint_dir.rstrip('/')), args.load_from)
        else:
            load_path = os.path.join(args.checkpoint_dir, args.name)
        old_config = vars(torch.load(os.path.join(load_path, f'{load_name}.pth')))
        old_config.update({k: v for k, v in config.items() if v is not None})
        config = old_config
    config['resume'] = resume

    return config

class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return pprint.pformat(vars(self))

def get_config():
    return Config(parse())
