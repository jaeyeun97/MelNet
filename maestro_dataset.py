"""
Maestro + GuitarSet Dataset class

For training a CycleGAN Network.
"""
from torch.util import data

import os
import pandas


class Maestro(data.Dataset):
    def __init__(self, opt, prefix):
        super(Maestro, self).__init__(self, opt, prefix)

        maestro_path = self.root
        maestro_name = os.path.basename(maestro_path)
        maestro_file = os.path.join(maestro_path, "{}.csv".format(maestro_name))
        maestro_meta = pandas.read_csv(maestro_file)

        splits = (maestro_meta.duration // self.duration).rename('splits')
        maestro_meta = maestro_meta.join(splits)[['audio_filename', 'splits']]
        self.paths = list()
        for index, row in maestro_meta.iterrows():
            for split in range(int(row.splits)):
                f = os.path.join(maestro_path, row.audio_filename)
                self.paths.append('{}:{}'.format(f, str(split)))
        self.size = len(self.paths)
 
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
        index -- a random integer for data indexing
        """

        path = self.paths[index]
        audio, split = tuple(path.split(':'))
        split = int(split)
        data = self.retrieve_audio(audio, split)

        return self.preprocess(data)

    def __len__(self):
        """Return the total number of audio files."""
        return min(len(self.paths), self.get_opt('max_dataset_size'))
