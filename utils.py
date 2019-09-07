import torch


def mdn_loss(mu, sigma, pi, target):
    dist = torch.distributions.normal.Normal(mu, sigma)
    return -dist.log_prob(target.unsqueeze(-1)).exp().mul(pi).sum(dim=-1).log().mean()


def sample(mu, sigma, pi):
    cat = torch.distributions.categorical.Categorical(pi) 
    dist = torch.distributions.normal.Normal(mu, sigma)
    idx = cat.sample().unsqueeze(-1)
    norms = dist.sample()
    return norms.gather(3, idx).squeeze(-1)


def split(x, axis):
    B, T, M = x.size()
    if axis:
        return x[:, 0::2, :], x[:, 1::2, :]
    else:
        return x[:, :, 0::2], x[:, :, 1::2]


def interleave(x, y):
    B, T, M = x.size()
    assert [B, T, M] == list(y.size())
    if T == M:
        new_tensor = x.new_empty((B, T, M*2))
        new_tensor[:, :, 0::2] = x
        new_tensor[:, :, 1::2] = y
        return new_tensor
    else:
        new_tensor = x.new_empty((B, T*2, M))
        new_tensor[:, 0::2, :] = x
        new_tensor[:, 1::2, :] = y
        return new_tensor

def get_splits(x, count):
    """ Does not include original x """
    B, T, M = x.size()
    if count == 2:
        return [(x,)]
    first, second = split(x, count % 2 != 0)  # first = x^{<g}, second = x^g
    splits = get_splits(first, count-1)
    splits.append((first, second))
    return splits


def generate_splits(x, count):
    """ Includes original x """
    B, T, M = x.size()
    yield x, x
    for i in range(count, 2, -1):
        x, y = split(x, i % 2 != 0)  # first = x^{<g}, second = x^g
        yield x, y
    yield x,

if __name__ == "__main__":
    n_layers = [12, 5, 4, 3, 2, 2]
    x = torch.ones(1, 256, 320)
    splits = generate_splits(x, len(n_layers))
    for s in splits:
        print(', '.join(str(t.size()) for t in s))
