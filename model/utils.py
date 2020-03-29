import torch

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def mdn_loss(mu, sigma, pi, target):
    log_probs = Normal(mu, sigma).log_prob(target.unsqueeze(-1))
    log_probs = torch.logsumexp(log_probs + pi, -1)
    return -log_probs.mean()

def sample_mdn(mu, sigma, pi):
    idx = Categorical(logits=pi).sample()
    return torch.normal(mu, sigma).gather(-1, idx.unsqueeze(-1)).squeeze(-1)

# axis: true if time, false if freq
def split(x, axis=True):
    B, T, M = x.size()
    if axis:
        return x[:, 0::2, :], x[:, 1::2, :]
    else:
        return x[:, :, 0::2], x[:, :, 1::2]


# Always interleave Time first
# axis false if freq true if time
def interleave(x, y, axis=True):
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
    return list(reversed(list(_split_gen(x, count))))


# Should always end up with time last
def _split_gen(x, count):
    """ Includes original x; outputs count - 1 pairs and 1 singleton """
    B, T, M = x.size()
    axis = count % 2 == 0  # True if odd, False if even
    for _ in range(count - 1):
        x, y = split(x, axis)  # first = x^{<g}, second = x^g
        yield x, y
        axis = not axis
    yield x


def get_div_factors(count):
    count = count - 1
    time, freq = count // 2, count // 2
    if count % 2 != 0:
        time += 1
    return time, freq
