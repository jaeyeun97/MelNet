import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.tensorboard.writer import figure_to_image


def is_bad_grad(grad):
    return grad.ne(grad).any() or grad.gt(1e6).any()


def get_grad_info(*networks):
    labels = list()
    # is_bads = list()
    avg_grads = list()
    max_grads = list()
    for network in networks:
        for n, p in network.named_parameters():
            if p.requires_grad and "bias" not in n:
                grad = p.grad
                labels.append(n)
                # is_bads.append(is_bad_grad(grad))
                grad = grad.abs()
                avg_grads.append(grad.mean().item())
                max_grads.append(grad.max().item())
    return labels, avg_grads, max_grads


def get_grad_plot(grad_info):
    labels, avg_grads, max_grads = grad_info
    fig = Figure(figsize=(15, 5))
    canvas = FigureCanvas(fig)
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
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    return np.moveaxis(image_hwc, source=2, destination=0)


def mdn_loss(mu, sigma, pi, target):
    dist = torch.distributions.normal.Normal(mu, sigma)
    logits = dist.log_prob(target.unsqueeze(-1))
    # probs = torch.exp(logits + pi)
    # probs = logits.exp().mul(pi)
    probs = torch.logsumexp(logits + pi, -1)
    return -probs.mean()


def sample(mu, sigma, pi):
    cat = torch.distributions.categorical.Categorical(logits=pi) 
    # cat = torch.distributions.categorical.Categorical(probs=pi) 
    dist = torch.distributions.normal.Normal(mu, sigma)
    idx = cat.sample().unsqueeze(-1)
    norms = dist.sample()
    return norms.gather(3, idx).squeeze(-1)


# Always split time first
# axis: true if time, false if freq
def split(x, axis=True):
    B, T, M = x.size()
    if axis:
        return x[:, 0::2, :], x[:, 1::2, :]
    else:
        return x[:, :, 0::2], x[:, :, 1::2]


# Always interleave freq first
# axis: true if time, false if freq
def interleave(x, y, axis=False):
    B, T, M = x.size()
    assert [B, T, M] == list(y.size())
    if axis:
        new_tensor = x.new_empty((B, T, M*2))
        new_tensor[:, :, 0::2] = x
        new_tensor[:, :, 1::2] = y
        return new_tensor
    else:
        new_tensor = x.new_empty((B, T*2, M))
        new_tensor[:, 0::2, :] = x
        new_tensor[:, 1::2, :] = y
        return new_tensor


def generate_splits(x, count):
    """ Includes original x; outputs count+1 pairs and 1 singleton """
    B, T, M = x.size()
    yield x, x
    axis = False
    for i in range(count, 0, -1):
        x, y = split(x, axis)  # first = x^{<g}, second = x^g
        yield x, y
        axis = not axis
    yield x


if __name__ == "__main__":
    import librosa
    n_layers = [12, 5, 4, 3, 2, 2]
    x, sr = librosa.load(librosa.util.example_audio_file())
    hop_length = 1536 // 4
    x = x[:255 * hop_length]
    x = librosa.feature.melspectrogram(x, sr, n_fft=1536, hop_length=hop_length, n_mels=256)
    x = torch.from_numpy(x)
    x = x.transpose(0, 1).unsqueeze(0)
    
    print(x.size())
    splits = list(generate_splits(x, len(n_layers) - 2))
    splits = list(reversed(splits))

    axis = False
    for i in range(1, len(splits) - 1):
        print(', '.join(str(t.size()) for t in splits[i]))
        x, y = splits[i]
        o = interleave(x, y, axis)
        assert (o - splits[i+1][0]).sum() == 0
        axis = not axis
