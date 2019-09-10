import torch


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
    n_layers = [12, 5, 4, 3, 2, 2]
    x = torch.ones(1, 256, 320)
    splits = generate_splits(x, len(n_layers) - 2)
    for s in splits:
        print(', '.join(str(t.size()) for t in s))
