from quantizer.kernel.util import (
    Boolean, Integer, Scalar, Type, beats2samples, samples2beats
)
from quantizer.kernel.waveform import Controller
import __main__
from abc import ABC, abstractmethod
import numpy as np


class Context:
    def __init__(
            self,
            beats: Scalar = 48,
            bpm: Scalar = 120,
            fs: Integer = 44100,
            nsamples: Integer = None,
            dtype: Type = np.float32,
    ):
        self.bpm = bpm
        self.fs = fs
        if nsamples:
            self.nsamples = nsamples
            self.beats = samples2beats(self.nsamples, self.bpm, self.fs)
        else:
            self.beats = beats
            self.nsamples = beats2samples(self.beats, self.bpm, self.fs)

        # axis
        x = np.arange(self.nsamples, dtype=dtype)
        self.x = Controller(x)

        # time
        t = np.arange(self.nsamples, dtype=dtype)
        t /= self.fs
        self.t = Controller(t)

        # frequency
        f = np.zeros(self.nsamples, dtype=dtype)
        f.fill(440)
        self.f = Controller(f)

        # phase
        p = np.zeros(self.nsamples, dtype=dtype)
        p.fill(0)
        self.p = Controller(p)

        # amplitude
        a = np.zeros(self.nsamples, dtype=dtype)
        a.fill(1)
        self.a = Controller(a)

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return (
            "QuantizerKernelContext {\n"
            f"\tbeats = {self.beats}\n"
            f"\tbpm = {self.bpm}\n"
            f"\tfs = {self.fs}\n"
            f"\tnsamples = {self.nsamples}\n"
            "}"
        )


class Session:

    def __init__(
            self,
            beats: Scalar = 48,
            bpm: Scalar = 120,
            fs: Integer = 44100,
            nsamples: Integer = None,
            fork: Boolean = True,
    ):
        if fork:
            setattr(
                __main__,
                "QUANTIZER_KERNEL_CONTEXT",
                Context(beats, bpm, fs, nsamples),
            )
        else:
            try:
                getattr(__main__, "QUANTIZER_KERNEL_CONTEXT")
            except AttributeError:
                setattr(
                    __main__,
                    "QUANTIZER_KERNEL_CONTEXT",
                    Context(beats, bpm, fs, nsamples),
                )

    @abstractmethod
    def session(self):
        raise RuntimeError


class Experiment(ABC):
    pass


Session(fork=False)
