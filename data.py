import os
import pandas
import librosa
import numpy as np

from absl import flags
from torch.utils import data


FLAGS = flags.FLAGS


def _partition(l, k, key):
    groups = [[] for i in range(k)]
    for i in sorted(l, key=key, reverse=True):
        groups.sort(key=lambda xs: sum(key(x) for x in xs))
        groups[0].append(i)
    return groups


def maestro(split):
    name = os.path.basename(FLAGS.dataroot)
    meta_file = os.path.join(FLAGS.dataroot, "{}.csv".format(name))
    meta = pandas.read_csv(meta_file)
    meta = meta.where(meta.split == split).dropna()
    audio_filenames = meta.audio_filename.map(lambda name: os.path.join(FLAGS.dataroot, name))
    num_frames = np.ceil(meta.duration * FLAGS.sample_rate / _get_frame_len()).astype(int)

    return pandas.DataFrame({
        'audio_filename': audio_filenames,
        'num_frames': num_frames
    }).itertuples()


class Dataset(data.IterableDataset):
    def __init__(self, split='train', world_size=1, rank=0):
        super(Dataset, self).__init__()

        if FLAGS.dataset == 'maestro':
            self.data = maestro(split)

        # num_workers = const * batch_size 
        self.world_size = world_size
        self.rank = rank
        self.worker_set = False

    def divide_dataset(self):
        if self.worker_set:
            self.data = _partition(self.data, FLAGS.num_workers,
                                  lambda entry: entry.num_frames)

    def set_worker(self, worker_id, num_workers):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.worker_set = True
        self.divide_dataset()

    def __iter__(self):
        frame_len = (FLAGS.nfft // 4) * FLAGS.timesteps - 1

        if self.worker_set:
            # each worker has their own version of the dataset
            data = self.data[self.rank * self.num_workers + self.worker_id]
        else:
            data = self.data

        for entry in data:
            sr = librosa.get_samplerate(entry.audio_filename)
            new_entry = True
            stream = librosa.stream(entry.audio_filename, 1,
                                    frame_len, frame_len,
                                    fill_value=0)
            for block in stream:
                yield block, new_entry
                new_entry = False

if __name__ == "__main__":
    import IPython
    from absl import app

    flags.DEFINE_string('dataroot', '../maestro-v2.0.0', 'Path to Maestro')
    flags.DEFINE_enum('dataset', 'maestro', ['maestro'], 'Which dataset')
    flags.DEFINE_integer('nfft', 2048, 'Number of STFT Frequency bins')
    flags.DEFINE_integer('timesteps', 256, 'Number of timesteps per data entry')
    flags.DEFINE_integer('sample_rate', 22050, 'Sample Rate')
    flags.DEFINE_integer('batch_size', 4, 'Batch Size')
    flags.DEFINE_integer('num_workers', 8, 'Num workers')

    def main(argv):
        IPython.embed()

    app.run(main)
