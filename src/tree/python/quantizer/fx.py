from quantizer.kernel.util import db2mag, qkc
from quantizer.kernel.waveform import Controller
from quantizer.stream import Stream
from abc import ABC, abstractmethod
from numba import jit
import numpy as np
from scipy.signal import fftconvolve, butter


class FX(ABC):
    @abstractmethod
    def patch(self, stream):
        """
        :type stream: Waveform
        :rtype: Stream
        """
        raise RuntimeError


class MonoFX(FX):
    pass


class StereoFX(FX):
    pass


class MultiFX(FX):
    pass


class TemporalFX(FX):
    pass


class SpectralFX(FX):
    pass


class TemporalMonoFX(MonoFX, TemporalFX):
    pass


class SpectralMonoFX(MonoFX, SpectralFX):
    pass


class Waveshaper(MonoFX):
    def __init__(self, gain=0, mode="tanh"):
        self.gain = Controller.cast(gain)
        self.mode = mode

    def patch(self, stream):
        s = db2mag(self.gain.get_ndarray()) * stream.get_ndarray()
        if self.mode == "erf":
            s = np.erf(s)
        elif self.mode == "tanh":
            s = np.tanh(s)
        elif self.mode == "arctan":
            s = 2 / np.pi * np.arctan(np.pi / 2 * s)
        elif self.mode == "clip":
            s1 = np.where(s < -1, -1, 0)
            s2 = np.where((s >= -1) & (s <= 1), s, 0)
            s3 = np.where(s > 1, 1, 0)
            s = s1 + s2 + s3
        else:
            raise RuntimeError
        return Stream(s)


class SincFilter(MonoFX):
    def __init__(self, f=440, bandwidth=440, mode="lp"):
        self.fc = f / qkc().fs * 2
        self.bw = bandwidth / qkc().fs
        self.m = int(np.round(4 / self.bw))
        if self.m % 2:
            self.n = self.m
        else:
            self.n = self.m + 1

    def patch(self, stream: Stream) -> Stream:
        s = stream.get_ndarray()
        x = np.arange(self.n)
        w = (
            0.42
            - 0.5 * np.cos(2 * np.pi * x / self.m)
            + 0.08 * np.cos(4 * np.pi * x / self.m)
        )
        h = np.sinc(self.fc * (x - self.m / 2))
        h *= w
        h /= np.sum(h)
        s = fftconvolve(s, h)
        return Stream(s)


class ButterworthFilter(MonoFX):
    def __init__(self, f=440, order=4):
        fc = f / qkc().fs * 2
        b, a = butter(order, fc)
        self.b = b
        self.a = a
        self.dx = [0.] * order
        self.dy = [0.] * order

    def patch(self, stream: Stream) -> Stream:
        x = stream.get_ndarray()
        y = np.zeros(len(x))
        y = self._patch(x, y, self.dx, self.dy, self.b, self.a)
        return Stream(y)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _patch(x, y, dx, dy, b, a):
        for i in range(len(x)):
            sigma_bx = b[0] * x[i]
            for j in range(1, len(b)):
                sigma_bx += b[j] * dx[j - 1]
            sigma_ay = 0.
            for j in range(1, len(a)):
                sigma_ay += a[j] * dy[j - 1]
            y[i] = (sigma_bx - sigma_ay) / a[0]
            for j in range(1, len(dx)):
                dx[-1 * j] = dx[-1 * j - 1]
            dx[0] = x[i]
            for j in range(1, len(dy)):
                dy[-1 * j] = dy[-1 * j - 1]
            dy[0] = y[i]
        return y


SF = SincFilter
BWF = ButterworthFilter
