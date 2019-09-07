import os
import argparse
import pprint
import torch

from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import Compose


def get_optimizer(optimizer_name='Adam'):
    """Get optimizer by name"""
    # optimizer_name = optimizer_name.capitalize()
    return getattr(torch.optim, optimizer_name)


class Config(object):
    def __init__(self):
        # TODO: from JSON file
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='maestro', help='Which dataset to use: [maestro | musicnet]')
        parser.add_argument('--dataroot', type=str, default='./dataset', help='parent directory of datasets')
        parser.add_argument('--mode', type=str, default='train', help='[train | val | test | sample]')
        
        parser.add_argument('--offload-dir', type=str, default=f'/tmp/{os.getpid()}/')
        parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint')

        # --- Audio --- #
        parser.add_argument('--sample-rate', type=int, default=22050)
        # TODO: implement CQT measures
        parser.add_argument('--spectrogram', type=str, default='mel_stft', help='[mel_stft | cqt]')

        opt, _ = parser.parse_known_args()

        # --- Mel-STFT specific --- #
        if opt.spectrogram == 'mel_stft':
            parser.add_argument('--n-fft', type=int, default=2048)
            parser.add_argument('--n-mels', type=int, default=256)
            parser.add_argument('--win-length', type=int, default=None)
            parser.add_argument('--hop-length', type=int, default=None)

        # --- Mode Specific --- #
        if opt.mode == 'train':
            parser.add_argument('--batch-size', type=int, default=1)
            parser.add_argument('--epochs', type=int, default=200)
            parser.add_argument('--optimizer', type=str, default='Adam')
            parser.add_argument('--lr', type=float, default=1e-3)

        # --- Network --- #
        # Network width
        parser.add_argument('--width', type=int, default=512)
        # Number of mixtures
        parser.add_argument('--mixtures', type=int, default=10)

        args = parser.parse_args()
        self.config = vars(args)
        self.initvars()

    def initvars(self):
        self.config['optimizer'] = get_optimizer(self.config['optimizer'])
        preprocesses = []

        if self.config['spectrogram'] == 'mel_stft':
            if self.config['win_length'] is None:
                self.config['win_length'] = self.config['n_fft']

            if self.config['hop_length'] is None:
                self.config['hop_length'] = self.config['n_fft'] // 4

            preprocesses.append(MelSpectrogram(sample_rate=self.config['sample_rate'],
                                               n_fft=self.config['n_fft'],
                                               n_mels=self.config['n_mels'],
                                               win_length=self.config['win_length'],
                                               hop_length=self.config['hop_length']))
        self.config['preprocess'] = Compose(preprocesses)

    def __getitem__(self, name):
        return self.config['name']

    def __setitem__(self, name, value):
        self.config[name] = value

    def __getattr__(self, name):
        return self.config['name']

    def __repr__(self):
        return pprint.pformat(self.config)


if __name__ == "__main__":
    config = Config()
    print(config)
