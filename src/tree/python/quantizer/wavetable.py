from quantizer.kernel.util import (
    Array, Integer, Scalar, midi2freq, normalize, qkc
)
from abc import ABC, abstractmethod
import numpy as np

np.random.seed(0)


class Wavetable(ABC):

    @staticmethod
    @abstractmethod
    def fn(x: Array) -> Array:
        raise RuntimeError

    def render(self, x: Array) -> Array:
        return self.fn(((x + np.pi) % (2 * np.pi)) - np.pi)

    def discretize(
        self,
        f: Scalar = midi2freq(60),
        fs: Integer = None,
    ) -> Array:
        if not fs:
            fs = qkc().fs
        window = round(fs / f)
        x = np.linspace(-np.pi, np.pi, window)
        y = self.fn(x)
        return y


class Sine(Wavetable):

    @staticmethod
    def fn(x: Array) -> Array:
        y = np.sin(x)
        return normalize(y)


class Square(Wavetable):

    @staticmethod
    def fn(x: Array) -> Array:
        y1 = np.where(x < 0, -1, 0)
        y2 = np.where(x > 0, 1, 0)
        y = y1 + y2
        return normalize(y)


class Saw(Wavetable):

    @staticmethod
    def fn(x: Array) -> Array:
        y = x
        return normalize(y)


class Triangle(Wavetable):

    @staticmethod
    def fn(x: Array) -> Array:
        y1 = np.where(
            x < (-np.pi / 2),
            (-2 / np.pi) * x - 2,
            0
        )
        y2 = np.where(
            (x >= (-np.pi / 2)) & (x <= (np.pi / 2)),
            (2 / np.pi) * x,
            0
        )
        y3 = np.where(
            x > (np.pi / 2),
            (-2 / np.pi) * x + 2,
            0
        )
        y = y1 + y2 + y3
        return normalize(y)


class Noise(Wavetable):

    @staticmethod
    def fn(x: Array) -> Array:
        y = np.random.uniform(-1, 1, len(x))
        return normalize(y)


Sin = Sine
Squ = Square
Tri = Triangle
