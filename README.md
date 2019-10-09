# MelNet Reimplementation

This is a PyTorch reimplementation of [MelNet](https://arxiv.org/abs/1906.01083).

Implementation seems to be reasonably working. I'm using 2x2080Ti (which was a huge investment on my part) to train the network. Even then, I did not have enough GPU memory to run the network with the hyperparameters the paper has set out to use. I've used NVIDIA's `apex` in order to fit wider networks into the GPUs, and I got reasonable results for:

Batchsize: 16, Width: 64, Hop Length: 256, Timesteps: 384, Sample Rate: 16384 (~5 seconds of audio).
Batchsize: 8, Width: 128, Hop Length: 256, Timesteps: 384, Sample Rate: 16384 (~5 seconds of audio).


### Dataset

This has been accommodated for the `maestro` dataset from Google only; please contribute if you are looking to implement the TTS version of MelNet.

### Parallelisation

This implementation utilises `apex.parallel.DistributedDataParallel` module and `torch.multiprocessing` in order to train on multiple GPUs. Please see the implementation details if you are looking for a way to run a PyTorch processes for each GPU.

### License

I haven't decided on the license of the code here. For now, CC BY-SA policy applies while I decide whether I want this to be GPL or MIT, etc.


### Misc.

I've invested quite a bit of time into hardware while writing this reimplementation; if you were to use it, please consider a donation. As always, contributions are welcome.
