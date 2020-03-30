import torch
import time

from .model import MelNet

batchsize = 1
timesteps = 256
n_mels = 256
width = 512
n_layers = 16
n_mixtures = 10

gpu = torch.device('cuda:1')
# model = InitialTier(width, n_mels, n_layers, n_mixtures).to(device=gpu)
# f_ext = FeatureExtraction(width).to(torch.half).to(gpu)
torch.cuda.set_device(gpu)

model = MelNet(width, n_mels, [16, 6, 5, 4], n_mixtures)
model.train()

print(model.state_dict())

x = torch.ones(batchsize, timesteps, n_mels, device=gpu)
# c = torch.zeros(batchsize, timesteps, n_mels, device=gpu)
model.zero_grad()
losses = model(x, [0], [False])
print(losses)
model.step()
# start_time = time.time()
# with torch.no_grad():
#     x = model.sample(timesteps)
#     print(x.size())
# print(f"---- {time.time() - start_time} ----")

# f_ext.train()
# print("Input Shape:", x.shape)
# c = f_ext(x)
# print("Conditional Shape", c.shape)
# mu, sigma, pi = model(x, ) # , c)
# print("Output Shape", mu.shape)
# 
# loss = mdn_loss(mu, sigma, pi, x)
# loss.backward()
# print(loss)
# print(sample(mu, sigma, pi).size())

# 64 -> 64 -> 128 -> 128 -> 256 -> 256 # -> 512 -> 512 
# 80 -> 80 -> 80  -> 160 -> 160 -> 320 # -> 320 -> 640
# 12 -> 5  -> 4   -> 3   -> 2   -> 2
