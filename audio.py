import torch
import librosa

from torchaudio.functional import create_fb_matrix

""" Reimplementations to use with CUDA operations """


class PowerToDB(object):
    def __init__(self, eps=1e-10, db_range=80.0, normalized=True):
        self.eps = eps
        self.db_range = db_range
        self.normalized = normalized

    def __call__(self, x):
        x = x.clamp(min=self.eps)  # new tensor here
        ref = x.max().log10().item()
        x = x.log10_().sub_(ref).mul_(10)
        x = x.clamp_(min=-self.db_range)

        if self.normalized:
            return (x + self.db_range) / self.db_range
        else:
            return x


class MelScale(object):
    def __init__(self, sample_rate=22050, n_fft=2048, n_mels=256,
                 f_min=0., f_max=None):
        f_max = float(sample_rate // 2) if f_max is None else f_max
        assert f_min <= f_max
        self.fb = create_fb_matrix(n_fft // 2 + 1, f_min, f_max, n_mels)

    def __call__(self, spec):
        self.fb = self.fb.to(dtype=spec.dtype, device=spec.device)
        return torch.matmul(spec.transpose(-1, -2), self.fb)


class Spectrogram(object):
    def __init__(self, n_fft=2048, win_length=None, hop_length=None,
                 window_fn=torch.hann_window, wkargs=None, normalized=False,
                 power=2., pad_mode='reflect'):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.power = power

        self.window = window_fn(self.win_length) if wkargs is None else window_fn(self.win_length, **wkargs)

    def __call__(self, x):
        self.window = self.window.to(dtype=x.dtype, device=x.device)
        x = x.stft(self.n_fft,
                   hop_length=self.hop_length,
                   win_length=self.win_length,
                   window=self.window,
                   pad_mode=self.pad_mode)

        if self.normalized:
            x /= self.window.pow(2).sum().sqrt()
        return x.pow_(self.power).sum(-1)


if __name__ == "__main__":
    import librosa.display
    # import numpy as np
    import matplotlib.pyplot as plt

    # plt.figure()

    x, sr = librosa.load(librosa.util.example_audio_file())

    # S = librosa.stft(x, n_fft=1536, hop_length=256)
    # print(np.max(S))
    # print(np.mean(S))
    # plt.subplot(4, 1, 1)
    # plt.hist(x)
    # plt.title('STFT Histogram')

    # S = np.abs(S)
    # print(np.max(S))
    # print(np.mean(S))
    # plt.subplot(4, 1, 2)
    # librosa.display.specshow(S, sr=sr)
    # plt.colorbar()
    # plt.title('Mag STFT Histogram')

    # S = librosa.feature.melspectrogram(sr=sr, S=S, n_fft=1536, hop_length=256, n_mels=256)
    # print(np.max(S))
    # print(np.mean(S))
    # plt.subplot(4, 1, 3)
    # librosa.display.specshow(S, sr=sr, y_axis='log')
    # plt.colorbar()
    # plt.title('Mel-Mag STFT Histogram')

    # S = librosa.power_to_db(S**2, ref=np.max)
    # print(np.max(S))
    # print(np.mean(S))
    # plt.subplot(4, 1, 4)
    # librosa.display.specshow(S, sr=sr, y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Mag-dB spectrogram')
    # plt.tight_layout()
    # plt.show()

    x = torch.from_numpy(x) # .to('cuda:0')
    melscale = MelScale(sample_rate=sr, n_fft=1536, n_mels=256)
    spectrogram = Spectrogram(n_fft=1536, hop_length=256, win_length=1536, power=2., normalized=True)
    logpower = PowerToDB(normalized=False)
    x = spectrogram(x)
    print(x.size())
    print(x.max())
    print(x.mean())
    x = melscale(x)
    print(x.size())
    print(x.max())
    print(x.mean())
    print(x.min())
    x = logpower(x)
    print(x.size())
    print(x.max())
    print(x.mean())
    print(x.min())
    # x = x.transpose(0, 1).numpy()
    # librosa.display.specshow(x, sr=sr, y_axis='log')
    # plt.show()
