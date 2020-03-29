import os
import pandas
import librosa
import numpy as np
import resampy
import math

from absl import flags
from itertools import chain
from torch.utils.data import IterableDataset

def _partition(l, k, key):
    if k == 1:
        return [l]
    groups = [[] for i in range(k)]
    for i in sorted(l, key=key, reverse=True):
        groups.sort(key=lambda xs: sum(key(x) for x in xs))
        groups[0].append(i)
    return groups

def maestro(dataroot, split, sample_rate, hop_length, timesteps, method, center, n_fft):
    name = os.path.basename(dataroot)
    meta_file = os.path.join(dataroot, "{}.csv".format(name))
    meta = pandas.read_csv(meta_file)
    meta = meta.where(meta.split == split).dropna()
    audio_filenames = meta.audio_filename.map(lambda name: os.path.join(dataroot, name))
    # num_samples = meta.duration * sample_rate
    # WARNING: meta.duration is based on MIDI not the actual file
    durations = audio_filenames.map(lambda f: librosa.get_duration(filename=f))
    num_samples = durations * sample_rate
    if method == 'cqt' or center:
        total_size = num_samples // hop_length + 1
    else:
        total_size = (num_samples - n_fft) // hop_length + 1
    num_frames = np.ceil(total_size / timesteps).astype(int)

    return pandas.DataFrame({
        'audio_filename': audio_filenames,
        'num_frames': num_frames
    }).itertuples()


class Dataset(IterableDataset):
    def __init__(self, dataset, dataroot, split='train', batch_size=1,
                 sample_rate=24000, timesteps=256, method='mel',
                 hop_length=256, center=False, n_fft=1536, n_mels=256, # as per paper
                 n_bins=336, bins_per_octave=48, num_workers=0, world_size=1):
        super(Dataset, self).__init__()

        if dataset == 'maestro':
            self.data = maestro(dataroot, split, sample_rate,
                                hop_length, timesteps, method,
                                center, n_fft)
        else:
            raise NotImplementedError()

        self.rank = 0
        self.worker_id = 0
        self.num_workers = num_workers
        self.world_size = world_size

        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.timesteps = timesteps
        self.method = method
        self.hop_length = hop_length

        if self.method == 'mel':
            self.center = center
            self.n_fft = n_fft
            self.n_mels = n_mels
        elif self.method == 'cqt':
            self.n_bins = self.n_bins
            self.bins_per_octave = bins_per_octave
        else:
            raise NotImplementedError

        if self.num_workers >= batch_size:
            # each worker works on one chunk at a time 
            chunk_size = self.world_size * self.num_workers
        else:
            # 1. synchronous dataloading: interleave data
            # 2. worker < batchsize: each worker generates interleaved data
            chunk_size = self.batch_size * max(self.num_workers, 1) * self.world_size
        self.data = _partition(self.data, chunk_size,
                               lambda entry: entry.num_frames)
        self.len = min(sum(entry.num_frames for entry in chunk) for chunk in self.data)

    def set_rank(self, rank):
        self.rank = rank

    def set_worker(self, worker_id):
        self.worker_id = worker_id

    def __iter__(self):
        if self.num_workers >= self.batch_size:
            entries = self.data[self.rank * self.num_workers + self.worker_id]
            yield from self._data_generator(entries)
        else:
            # batch_size * num_workers * world_size chunks into batch_size slices per worker
            i = self.batch_size * (self.rank * max(1, self.num_workers) + self.worker_id)
            generators = [self._data_generator(chunk) for chunk in self.data[i:i+self.batch_size]]
            yield from chain.from_iterable(zip(*generators))

    def _data_generator(self, entries):
        count = 0
        for entry in entries:
            sr = librosa.get_samplerate(entry.audio_filename)

            if self.method == 'mel' and not self.center and sr % self.sample_rate == 0:
                stream = self._stream_generator(entry.audio_filename, sr)
            else:
                stream = self._block_generator(entry.audio_filename, sr, entry.num_frames)

            for i, block in enumerate(stream):
                if count >= self.len:
                    return
                block = librosa.power_to_db(block, ref=np.max, top_db=80.0)
                block = block / 80 + 1
                yield count, entry.Index, block, (i == entry.num_frames - 1)
                count += 1

    def _stream_generator(self, filename, sr):
        "Used for generating Mel Spectrograms when given sample rate is an integer multiple of the original"
        assert sr % self.sample_rate == 0, f'Sample Rate {sr} is not a multiple, use block generator instead'
        mult = sr // self.sample_rate
        stream = librosa.stream(filename,
                                block_length=self.timesteps,
                                frame_length=self.n_fft * mult,
                                hop_length=self.hop_length * mult,
                                fill_value=0)

        for block in stream:
            yield librosa.feature.melspectrogram(block, sr=sr,
                                                 n_fft=self.n_fft * mult,
                                                 hop_length=self.hop_length * mult,
                                                 n_mels=self.n_mels,
                                                 center=False)

    def _block_generator(self, filename, sr, num_frames):
        "Default block generator"
        if sr % self.sample_rate == 0:
            y, _ = librosa.load(filename, sr=None)
            mult = sr // self.sample_rate
        else:
            y, sr = librosa.load(filename, sr=self.sample_rate)
            mult = 1

        # Pad for equal length
        total_len = num_frames * self.timesteps * self.hop_length * mult - 1
        if self.method == 'mel' and not self.center:
            total_len += self.n_fft * mult
        y = np.pad(y, (0, total_len - y.shape[-1]))

        if self.method == 'mel':
            S = librosa.feature.melspectrogram(y, sr=sr,
                                               n_fft=self.n_fft * mult,
                                               hop_length=self.hop_length * mult,
                                               n_mels=self.n_mels,
                                               center=self.center)
        elif self.method == 'cqt':
            S = librosa.cqt(y, sr=sr,
                            hop_length=self.hop_length * mult,
                            n_bins=self.n_bins,
                            bins_per_octave=self.bins_per_octave)
            S = np.abs(S) ** 2

        assert S.shape[-1] % self.timesteps == 0

        for i in range(0, S.shape[-1], self.timesteps):
            yield S[:, i:i+self.timesteps]
