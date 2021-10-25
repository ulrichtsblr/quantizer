import numpy as np
from typing import Union


def minmaxscale_i16(stream: np.ndarray) -> np.ndarray:
    """
    Arbitrary min-max scaling from numpy.float64 to numpy.int16.
    https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)

    :param stream: initial audio data
    :return: scaled audio data
    """
    a0, b0 = -1, 1
    a1, b1 = -32768, 32767
    stream = a1 + (stream - a0) * (b1 - a1) / (b0 - a0)
    return stream.astype(np.int16)


def midi2freq(m: int) -> float:
    """
    MIDI number to frequency conversion.
    https://newt.phys.unsw.edu.au/jw/notes.html

    :param m: midi number
    :return: frequency
    """
    if m == 0:
        return 0
    else:
        return 2 ** ((m - 69) / 12) * 440


def beats2samples(n: Union[int, float], bpm: int, fs: int) -> int:
    """
    Conversion of number of beats (quarter notes) to number of samples.

    :param n: number of beats
    :param bpm: beats per minute
    :param fs: sampling frequency
    :return: number of samples
    """
    return np.round(n * (1 / bpm) * 60 * fs).astype(int)
