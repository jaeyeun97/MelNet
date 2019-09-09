import torch
from torchaudio.functional import create_fb_matrix, complex_norm

""" Reimplementations to use with CUDA operations """


class MelScale(object):
    def __init__(self, sample_rate=22050, n_fft=2048, n_mels=256,
                 f_min=0., f_max=None, dtype=None, device=None):
        f_max = float(sample_rate // 2) if f_max is None else f_max
        assert f_min <= f_max
        fb = create_fb_matrix(n_fft // 2 + 1, f_min, f_max, n_mels)
        self.fb = fb.to(dtype=dtype, device=device)

    def __call__(self, specgram):
        return torch.matmul(specgram.transpose(-1, -2), self.fb)


class Spectrogram(object):
    def __init__(self, n_fft=2048, win_length=None, hop_length=None,
                 window_fn=torch.hann_window, wkargs=None, normalized=False,
                 power=2., pad_mode='reflect', dtype=None, device=None):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.power = power
        self.device = device
        self.dtype = dtype

        window = window_fn(self.win_length) if wkargs is None else window_fn(self.win_length, **wkargs)
        self.window = window.to(dtype=dtype, device=device)

    def __call__(self, x):
        x = x.stft(self.n_fft, hop_length=self.hop_length,
                   win_length=self.win_length, window=self.window,
                   pad_mode=self.pad_mode)

        if self.normalized:
            x /= self.window.pow(2).sum().sqrt()
        return complex_norm(x, power=self.power)


if __name__ == "__main__":
    import librosa
    x, sr = librosa.load(librosa.util.example_audio_file())
    x = torch.from_numpy(x).to('cuda:0')
    melscale = MelScale(sample_rate=sr, n_fft=1536, n_mels=256, device='cuda:0')
    spectrogram = Spectrogram(n_fft=1536, hop_length=256, win_length=1536, device='cuda:0')
    x = spectrogram(x)
    x = melscale(x)
    print(x.size())
