import torch
import math

from torchaudio.functional import create_fb_matrix, istft, complex_norm
from torchvision.transforms import Compose

""" Reimplementations to use with CUDA operations """


def get_audio_processes(config):
    preprocess = Compose([
        ToDevice(config.preprocess_device),
        Spectrogram(n_fft=config.n_fft,
                    win_length=config.win_length,
                    hop_length=config.hop_length,
                    power=1),
        MelScale(sample_rate=config.sample_rate,
                 n_fft=config.n_fft,
                 n_mels=config.n_mels),
        LogAmplitude()
    ])
    postprocess = Compose([
        LinearAmplitude(),
        MelToLinear(sample_rate=config.sample_rate,
                    n_fft=config.n_fft,
                    n_mels=config.n_mels),
        InverseSpectrogram(n_fft=config.n_fft,
                           win_length=config.win_length,
                           hop_length=config.hop_length,
                           length=config.frame_length,
                           power=1)
    ])
    return preprocess, postprocess


class ToDevice(object):
    def __init__(self, device):
        if device == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', device)

    def __call__(self, x):
        return x.to(self.device)


class LogAmplitude(object):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def __call__(self, x):
        return x.add(self.eps).log()


class LinearAmplitude(object):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def __call__(self, x):
        return x.exp().sub(self.eps)


class MelScale(object):
    def __init__(self, sample_rate=22050, n_fft=2048, n_mels=256,
                 f_min=0., f_max=None):
        f_max = float(sample_rate // 2) if f_max is None else f_max
        assert f_min <= f_max
        self.fb = create_fb_matrix(n_fft // 2 + 1, f_min, f_max, n_mels)

    def __call__(self, spec):
        self.fb = self.fb.to(dtype=spec.dtype, device=spec.device)
        return torch.matmul(spec.transpose(-1, -2), self.fb)


class MelToLinear(object):
    def __init__(self, sample_rate=22050, n_fft=2048, n_mels=256,
                 f_min=0., f_max=None):
        f_max = float(sample_rate // 2) if f_max is None else f_max
        assert f_min <= f_max
        # freq, mel -> mel, freq
        self.fb = create_fb_matrix(n_fft // 2 + 1, f_min, f_max, n_mels)

    def __call__(self, melspec):
        self.fb = self.fb.to(dtype=melspec.dtype, device=melspec.device)
        if len(melspec.size()) < 3:
            melspec = melspec.unsqueeze(0)

        n, m = self.fb.size()  # freq(n) x mel(m)
        b, k, m2 = melspec.size()  # b x time(k) x mel(m)
        assert m == m2
        X = torch.zeros(b, k, n, requires_grad=True,
                        dtype=melspec.dtype, device=melspec.device)
        optim = torch.optim.LBFGS([X])

        for _ in range(m):
            def step_func():
                optim.zero_grad()
                diff = melspec - X.matmul(self.fb)
                loss = diff.pow(2).sum().mul(0.5)
                loss.backward()
                return loss
            optim.step(step_func)

        X.requires_grad_(False)
        return X.clamp(min=0).transpose(-1, -2)


class Spectrogram(object):
    def __init__(self, n_fft=2048, win_length=None, hop_length=None,
                 window_fn=torch.hann_window, wkargs=None, normalized=False,
                 power=2., pad_mode='reflect'):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
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
        return complex_norm(x).pow_(self.power)


class InverseSpectrogram(object):
    """Adaptation of the `librosa` implementation"""
    def __init__(self, n_fft=2048, n_iter=32, hop_length=None, win_length=None,
                 window_fn=torch.hann_window, wkargs=None, normalized=False,
                 power=2., pad_mode='reflect', length=None, momentum=0.99):

        assert momentum < 1, f'momentum={momentum} > 1 can be unstable'
        assert momentum > 0, f'momentum={momentum} < 0'

        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.window = window_fn(self.win_length) if wkargs is None else window_fn(self.win_length, **wkargs)
        self.normalized = normalized
        self.pad_mode = pad_mode
        self.length = length
        self.power = power
        self.momentum = momentum / (1 + momentum)

    def __call__(self, S):
        self.window = self.window.to(dtype=S.dtype, device=S.device)

        S = S.pow(1/self.power)
        if self.normalized:
            S *= self.window.pow(2).sum().sqrt()

        # randomly initialize the phase
        angles = 2 * math.pi * torch.rand(*S.size())
        angles = torch.stack([angles.cos(), angles.sin()], dim=-1).to(dtype=S.dtype, device=S.device)
        S = S.unsqueeze(-1).expand_as(angles)

        # And initialize the previous iterate to 0
        rebuilt = 0.

        for _ in range(self.n_iter):
            # Store the previous iterate
            tprev = rebuilt

            # Invert with our current estimate of the phases
            inverse = istft(S * angles,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window,
                            length=self.length)

            # Rebuild the spectrogram
            rebuilt = inverse.stft(n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length,
                                   window=self.window, pad_mode=self.pad_mode)

            # Update our phase estimates
            angles = rebuilt.sub(self.momentum).mul_(tprev)
            angles = angles.div_(complex_norm(angles).add_(1e-16).unsqueeze(-1).expand_as(angles))

        # Return the final phase estimates
        return istft(S * angles,
                     n_fft=self.n_fft,
                     hop_length=self.hop_length,
                     win_length=self.win_length,
                     window=self.window,
                     length=self.length)


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt
    x, sr = librosa.load(librosa.util.example_audio_file())
    l = len(x)

    x = torch.from_numpy(x).to('cuda:0')
    melscale = MelScale(sample_rate=sr, n_fft=1536, n_mels=256)
    spectrogram = Spectrogram(n_fft=1536, hop_length=256, win_length=1536, power=1)
    logamp = LogAmplitude()
    X = spectrogram(x)
    Y = melscale(X)
    Z = logamp(Y)
    print(Z.min())
    print(Z.max())
    print(Z.mean())
    print(Z.var())
    Z = Z.unsqueeze(0)
    linearamp = LinearAmplitude()
    meltolin = MelToLinear(sample_rate=sr, n_fft=1536, n_mels=256)
    ispec = InverseSpectrogram(n_fft=1536, hop_length=256, win_length=1536, power=1, length=l)
    Y_hat = linearamp(Z)
    print(f'dB Error: {(Y_hat - Y).pow(2).mean()}')
    X_hat = meltolin(Y_hat)
    print(f'Mel Error: {(X - X_hat).pow(2).mean()}')
    print(X_hat.size())
    x_hat = ispec(X_hat)
    print(f'Spec Error: {(x_hat - x).pow(2).mean()}')
    for i in range(x_hat.size(0)):
        librosa.output.write_wav(f'/home/jaeyeun/test_griffinlim_{i}.wav', x_hat[i, :].cpu().numpy(), sr=sr)
