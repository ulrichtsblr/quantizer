import __main__
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
from typing import Callable, Union
from warnings import warn

PI = np.pi

Boolean = bool
Integer = int
Float = float
Complex = complex
Character = str
String = str
Tuple = tuple
List = list
Set = set
Dict = dict
Array = np.ndarray
Scalar = (int, float)
Sequence = (tuple, list, set)
Callable = Callable
Type = type


def beats2samples(
        beats: Scalar,
        bpm: Scalar,
        fs: Integer,
) -> Integer:
    """
    Conversion of number of beats (quarter notes) to number of samples.

    :param beats: number of beats
    :param bpm: beats per minute
    :param fs: sampling frequency
    :return: number of samples
    """
    return np.round(beats * (1 / bpm) * 60 * fs).astype(int)


def samples2beats(nsamples: Integer, bpm: Scalar, fs: Integer) -> Float:
    """
    Conversion of number of samples to number of beats (quarter notes).

    :param nsamples: number of samples
    :param bpm: beats per minute
    :param fs: sampling frequency
    :return: number of beats
    """
    return nsamples * bpm * (1 / 60) * (1 / fs)


def bounce(y: Array, file_path: String = None, fs: Integer = None) -> None:
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    """
    if not file_path:
        file_path = "bounce.wav"
        warn(
            "quantizer.util.bounce was called with default "
            f"file_path = {file_path}",
            RuntimeWarning
        )
    if not fs:
        fs = 44100
        warn(
            "quantizer.util.bounce was called with default "
            f"fs = {fs}",
            RuntimeWarning
        )
    wavfile.write(file_path, fs, y.astype(np.float32))
    return None


def broadcast(y: Array, fs: Integer = None) -> None:
    if not fs:
        fs = 44100
        warn(
            "quantizer.util.broadcast was called with default "
            f"fs = {fs}",
            RuntimeWarning
        )
    sd.play(y, fs)
    sd.wait()
    sd.stop()


def mag2db(mag: Float) -> Float:
    """
    https://www.mathworks.com/help/signal/ref/mag2db.html
    """
    return 20 * np.log10(mag)


def db2mag(db: Float) -> Float:
    return 10 ** (db / 20)


def midi2freq(m: Integer) -> Float:
    """
    Rectified MIDI number to frequency conversion.
    https://newt.phys.unsw.edu.au/jw/notes.html

    :param m: midi number [0 - 127; ~]
    :return: frequency [0 - +Inf; Hz]
    """
    if m < 0:
        return 0
    else:
        return 2 ** ((m - 69) / 12) * 440


def midi2mag(m: Integer) -> Float:
    """
    Linear midi velocity to magnitude conversion.
    :param m: midi velocity [0 - 127; ~]
    :return: magnitude [0 - 1; ~]
    """
    return m / 127


def minmaxscale(
        stream: Array, a0: Scalar, b0: Scalar, a1: Scalar, b1: Scalar
) -> Array:
    """
    https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    """
    stream = a1 + (stream - a0) * (b1 - a1) / (b0 - a0)
    return stream


def minmaxscale_i16(stream: Array) -> Array:
    return minmaxscale(
        stream,
        a0=-1,
        b0=1,
        a1=-32768,
        b1=32767,
    ).astype(np.int16)


def normalize(stream: Array) -> Array:
    return minmaxscale(
        stream,
        a0=np.min(stream),
        b0=np.max(stream),
        a1=-1,
        b1=1,
    )


def spn2midi(
        note: Character,
        octave: Integer,
        sharp: Boolean = False,
        flat: Boolean = False,
) -> Integer:
    """
    Scientific pitch notation to MIDI number conversion.
    https://en.wikipedia.org/wiki/Scientific_pitch_notation
    :param note: note (e.g. "A")
    :param octave: octave (e.g. 4)
    :param sharp
    :param flat
    """
    root = {
        "c": 0,
        "d": 2,
        "e": 4,
        "f": 5,
        "g": 7,
        "a": 9,
        "b": 11
    }
    return root[note.lower()] + (octave + 1) * 12 + (sharp * 1) + (flat * -1)


def transpose(
        f: (Scalar, Array),
        oc: (Scalar, Array) = 0,
        st: (Scalar, Array) = 0,
        ct: (Scalar, Array) = 0
) -> Union[Float, Array]:
    """
    :param f: frequency
    :param oc: octaves
    :param st: semitones
    :param ct: cents
    :return: transposed frequency
    """
    c = 1200 * oc + 100 * st + 1 * ct
    return f * 2 ** (c / 1200)


def qkc():
    return getattr(__main__, "QUANTIZER_KERNEL_CONTEXT")
