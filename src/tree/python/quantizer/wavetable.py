from quantizer.utils import minmaxscale_i16, midi2freq
import numpy as np
from scipy.io import wavfile
from typing import Callable


class Wavetable:

    def __init__(self, fs: int = 44100, window: int = 1024, c: int = 60):
        self.fs = fs
        self.window = window
        self.c = c
        self.x = np.linspace(0, 2 * np.pi, self.window)
        self.y = None

    def get_fs(self):
        return self.fs

    def osc(self, f: Callable):
        self.y = f(self.x)

    def read(self, f: float, nsamples: int) -> np.ndarray:
        phase_idx = (
            np.round(self.window * f * np.arange(0, nsamples) / self.fs) % self.window
        ).astype(int)
        return self.y[phase_idx]

    def write(self, file_name: str = "wavetable.wav", segmented: bool = False):

        def segment(y: np.ndarray) -> np.ndarray:
            f = midi2freq(m=self.c)
            t = np.round((1 / f) * self.fs).astype(int)
            segment_idx = np.round(np.linspace(0, self.window - 1, t)).astype(int)
            return y[segment_idx]

        if segmented:
            wavfile.write(file_name, self.fs, minmaxscale_i16(stream=segment(self.y)))
        else:
            wavfile.write(file_name, self.fs, minmaxscale_i16(stream=self.y))


class Sine(Wavetable):

    def __init__(self):
        super().__init__()
        super().osc(lambda x: np.sin(x))


class RectifiedSine(Wavetable):

    def __init__(self):
        super().__init__()
        super().osc(lambda x: 2 * abs(np.sin(x)) - 1)


class Saw(Wavetable):

    def __init__(self):
        super().__init__()
        super().osc(lambda x: np.linspace(-1, 1, len(x)))


class Square(Wavetable):

    def __init__(self):
        super().__init__()
        super().osc(
            lambda x:
            np.concatenate(
                [np.ones(int(len(x) / 2)), -1 * np.ones(int(len(x) / 2))],
                axis=None
            )
        )


class Triangle(Wavetable):

    def __init__(self):
        super().__init__()
        super().osc(
            lambda x:
            np.concatenate(
                [np.linspace(-1, 1, int(len(x) / 2)), np.linspace(1, -1, int(len(x) / 2))],
                axis=None
            )
        )


class NoiseUniform(Wavetable):

    def __init__(self):
        super().__init__()
        super().osc(lambda x: np.random.uniform(-1, 1, len(x)))
