import numpy as np
import librosa
import librosa.display

from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def figure_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    return np.moveaxis(image_hwc, source=2, destination=0)


def spec_to_image(spec, **kwargs):
    fig = Figure()
    ax = fig.gca()
    librosa.display.specshow(spec, ax=ax, x_axis='time',
                             y_axis=kwargs['spec_type'],
                             hop_length=kwargs['hop_length'],
                             sr=kwargs['sample_rate'])
    return figure_to_image(fig)

def spec_to_audio(spec, **kwargs):
    if kwargs['spec_type'] == 'mel':
        return librosa.feature.inverse.mel_to_audio(spec,
                                                    sr=kwargs['sample_rate'],
                                                    n_fft=kwargs['n_fft'],
                                                    n_mels=kwargs['n_mels'],
                                                    hop_length=kwargs['hop_length'],
                                                    center=kwargs['center'])
    elif kwargs['spec_type'] == 'cqt':
        return librosa.griffinlim_cqt(spec,
                                      sr=kwargs['sample_rate'],
                                      hop_length=kwargs['hop_length'],
                                      bins_per_octave=kwargs['bins_per_octave'])
