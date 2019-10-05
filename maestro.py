import os
import pandas
import numpy as np
import soundfile as sf
import resampy
import torch

from torch.utils import data
from torch.multiprocessing import Manager


class Maestro(data.Dataset):
    def __init__(self, root, frame_length,
                 sample_rate=22050, split='train', size=4000,
                 manager=None, preprocess=None):
        super().__init__()

        name = os.path.basename(root)
        meta_file = os.path.join(root, "{}.csv".format(name))
        meta = pandas.read_csv(meta_file)

        meta = meta.where(meta.split == split).dropna()
        audio_filenames = meta.audio_filename.map(lambda name: os.path.join(root, name))
        # Do not use data that is mainly silence
        num_frames = np.ceil(meta.duration * sample_rate / frame_length).astype(int)

        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.preprocess = preprocess
        self.duration = self.frame_length / self.sample_rate
        self.meta = pandas.DataFrame({
            'filename': audio_filenames,
            'frame_nums': num_frames,
            'frame_cumsum': num_frames.cumsum().shift(1, fill_value=0)
        })

        self.size = min(num_frames.sum(), size)
        # meta['filename'] = audio_filenames
        # meta['frame_nums'] = num_frames
        # meta['frame_cumsum'] = num_frames.cumsum()
        # self.meta = meta

        # if manager is not None:
        #     self.cache = manager.dict()
        #     self.counter = manager.dict()
        # else:
        #     self.cache = {}
        #     self.counter = {}

    def __getitem__(self, index):
        idx = self.meta.where(self.meta.frame_cumsum <= index).last_valid_index()
        record = self.meta.loc[idx]
        offset = record.frame_cumsum

        start = (index - offset) * self.frame_length / self.sample_rate
        # end = start + self.frame_length

        # from librosa.load
        with sf.SoundFile(record.filename) as f:
            sr = f.samplerate
            f.seek(int(start * sr))
            y = f.read(frames=int(self.duration * sr),
                       dtype=np.float32, always_2d=True).T

        # mono
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # resample
        if sr != self.sample_rate:
            y = resampy.resample(y, sr, self.sample_rate, filter='kaiser_best')

        # length normalize
        n = y.shape[-1]
        if n > self.frame_length:
            y = y[:self.frame_length]
        elif n < self.frame_length:
            y = np.pad(y, (0, self.frame_length - n))

        if self.preprocess is not None:
            y = torch.from_numpy(y)
            y = self.preprocess(y)

        return y

    def __len__(self):
        return self.size


if __name__ == "__main__":
    import librosa
    manager = Manager()
    dataset = Maestro('../maestro-v2.0.0', 319 * 1025, sample_rate=22050, split='train', manager=manager)

    l = []
    for i in range(5):
        l.append(dataset[i])
    d1 = np.concatenate(l)
    record = dataset.meta.iloc[0]
    d2, sr = librosa.load(record.filename, offset=0, duration=dataset.duration*5)
    print(np.power(d1 - d2, 2).mean())
    print(dataset[record.frame_cumsum])
    print(librosa.load(dataset.meta.iloc[1].filename, offset=0, duration=dataset.duration))
    print(dataset[570])
