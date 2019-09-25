import os
import pandas
import numpy as np
import librosa
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

        y, _ = librosa.load(record.filename, sr=self.sample_rate,
                            offset=start,
                            duration=self.duration)
        y = librosa.util.fix_length(y, self.frame_length)

        if self.preprocess is not None:
            y = torch.from_numpy(y)
            y = self.preprocess(y)

        return y

        # if idx not in self.cache.keys():
        #     # open file, resample, and store the numpy array
        #     self.counter[idx] = 1
        #     self.cache[idx], _ = librosa.load(record.filename, sr=self.sample_rate)

        # data = self.cache[idx][start:end]

        # self.counter[idx] += 1
        # if self.counter[idx] == record.frame_nums:
        #     del self.cache[idx]
        #     del self.counter[idx]

        # return data

    def __len__(self):
        return self.size


if __name__ == "__main__":
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
