import IPython
import numpy as np
from absl import app, flags
from .dataset import Dataset
from .dataloader import DataLoader

FLAGS = flags.FLAGS

flags.DEFINE_string('dataroot', '/home/jaeyeun/KAIST/maestro-v2.0.0', 'Path to Maestro')
flags.DEFINE_enum('dataset', 'maestro', ['maestro'], 'Which dataset')
flags.DEFINE_integer('nfft', 2048, 'Number of STFT Frequency bins')
flags.DEFINE_integer('timesteps', 320, 'Number of timesteps per data entry')
flags.DEFINE_integer('sample_rate', 22050, 'Sample Rate')
flags.DEFINE_integer('batch_size', 4, 'Batch Size')
flags.DEFINE_integer('num_workers', 8, 'Num workers')

def main(argv):
    dataset = Dataset(FLAGS.dataset, FLAGS.dataroot, split='train',
                      batch_size=FLAGS.batch_size, sample_rate=FLAGS.sample_rate,
                      num_workers=FLAGS.num_workers, center=False,
                      timesteps=FLAGS.timesteps)
    dataloader = DataLoader(dataset, FLAGS.num_workers, FLAGS.batch_size)
    for data in dataloader:
        print(data[0], data[1].size(), data[2])
    IPython.embed()


app.run(main)
