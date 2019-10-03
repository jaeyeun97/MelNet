import torch
import numpy as np
import librosa.display

from torch.distributions.normal import Normal
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def mdn_loss(mu, sigma, pi, target):
    log_probs = Normal(mu, sigma).log_prob(target.unsqueeze(-1))
    log_probs = torch.logsumexp(log_probs + pi, -1)
    return -log_probs.mean()


# Should always end up with freq last
# axis: true if time, false if freq
def split(x, axis=True):
    B, T, M = x.size()
    if axis:
        return x[:, 0::2, :], x[:, 1::2, :]
    else:
        return x[:, :, 0::2], x[:, :, 1::2]


# Always interleave Freq first
# axis false if freq true if time
def interleave(x, y, axis=False):
    B, T, M = x.size()
    assert [B, T, M] == list(y.size())
    if axis:
        # Interleaving Time
        new_tensor = x.new_empty((B, T*2, M))
        new_tensor[:, 0::2, :] = x
        new_tensor[:, 1::2, :] = y
        return new_tensor
    else:
        # Interleaving Mel
        new_tensor = x.new_empty((B, T, M*2))
        new_tensor[:, :, 0::2] = x
        new_tensor[:, :, 1::2] = y
        return new_tensor


def generate_splits(x, count):
    """ Includes original x; outputs count pairs and 1 singleton """
    B, T, M = x.size()
    # yield x, x
    # count = 3 -> Mel first
    axis = True if count % 2 == 0 else False
    for i in range(count, 0, -1):
        x, y = split(x, axis)  # first = x^{<g}, second = x^g
        yield x, y
        axis = not axis
    yield x


def clip_grad(clip_size):
    return lambda h: h.clamp(min=-clip_size, max=clip_size)


def scale_grad(clip_size):
    clip_size = float(clip_size)

    def scale(grad):
        c = clip_size / (grad.norm(2).item() ** 2 + 1e-6)
        if c < 1:
            return grad.mul(c)
        else:
            return grad

    return scale


def is_bad_grad(grad):
    return grad.ne(grad).any() or grad.gt(1e6).any()


def get_grad_info(*networks):
    labels = list()
    avg_grads = list()
    max_grads = list()
    for network in networks:
        for n, p in network.named_parameters():
            if p.requires_grad and "bias" not in n:
                labels.append(n)
                grad = p.grad.abs()
                avg_grads.append(grad.mean().item())
                max_grads.append(grad.max().item())
    return labels, avg_grads, max_grads


def figure_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    return np.moveaxis(image_hwc, source=2, destination=0)


def get_grad_plot(grad_info):
    labels, avg_grads, max_grads = grad_info
    fig = Figure(figsize=(14, 9), dpi=96)
    ax = fig.gca()
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), avg_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(avg_grads)+1, lw=2, color="k")
    ax.set_xticks(range(0, len(avg_grads), 1))
    ax.set_xticklabels(labels, rotation="vertical")
    ax.set_xlim(left=0, right=len(avg_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_yscale('log')
    ax.set_xlabel("Layers")
    ax.set_ylabel("Average Gradient")
    # ax.grid(True)
    ax.legend([Line2D([0], [0], color="c", lw=4),
               Line2D([0], [0], color="b", lw=4),
               Line2D([0], [0], color="k", lw=4)],
              ['max-gradient', 'mean-gradient', 'zero-gradient']) 
    fig.subplots_adjust(bottom=0.4)
    return figure_to_image(fig)


def get_spectrogram(sample, hop_length=256, sr=22050):
    fig = Figure()
    ax = fig.gca()
    if len(sample.size()) > 2:
        sample = sample.squeeze(0)
    sample = sample.cpu().transpose(0, 1).numpy()
    librosa.display.specshow(sample, x_axis='time', y_axis='mel', ax=ax,
                             hop_length=hop_length, sr=sr)
    return figure_to_image(fig)


if __name__ == "__main__":
    import librosa
    n_layers = [14, 7, 6, 5, 2]
    x, sr = librosa.load(librosa.util.example_audio_file())
    hop_length = 1536 // 4
    x = x[:255 * hop_length]
    x = librosa.feature.melspectrogram(x, sr, n_fft=1536, hop_length=hop_length, n_mels=256)
    x = torch.from_numpy(x)
    x = x.transpose(0, 1).unsqueeze(0)

    splits = list(generate_splits(x, len(n_layers) - 2))

    for i in range(len(splits) - 1):
        print(', '.join(str(t.size()) for t in splits[i]))
    print(splits[-1].size())

    splits = list(reversed(splits))
    axis = True
    for i in range(1, len(splits) - 1):
        print(', '.join(str(t.size()) for t in splits[i]))
        x, y = splits[i]
        o = interleave(x, y, axis)
        assert (o - splits[i+1][0]).sum() == 0
        axis = not axis
