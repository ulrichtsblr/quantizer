from quantizer.kernel.util import (
    Array, Integer, Scalar, String, broadcast, bounce, qkc
)
import numpy as np


class Waveform:
    def __init__(self, ndarray: Array = None):
        self._ndarray = ndarray

    def get_ndarray(self):
        return self._ndarray

    def set_ndarray(self, ndarray):
        self._ndarray = ndarray

    def add(self, other):
        if isinstance(other, Scalar):
            return Waveform(self._ndarray + other)
        elif isinstance(other, Array):
            return Waveform(self._ndarray + other)
        elif isinstance(other, Waveform):
            return Waveform(self._ndarray + other.get_ndarray())
        else:
            raise RuntimeError

    def mul(self, other):
        if isinstance(other, Scalar):
            return Waveform(self._ndarray * other)
        elif isinstance(other, Array):
            return Waveform(self._ndarray * other)
        elif isinstance(other, Waveform):
            return Waveform(self._ndarray * other.get_ndarray())
        else:
            raise RuntimeError

    def matmul(self, other, r):
        if isinstance(other, Integer):
            return Waveform(np.tile(self._ndarray, other))
        elif isinstance(other, Array):
            if r:
                return Waveform(np.concatenate([other, self._ndarray]))
            else:
                return Waveform(np.concatenate([self._ndarray, other]))
        elif isinstance(other, Waveform):
            if r:
                return Waveform(
                    np.concatenate([other.get_ndarray(), self._ndarray])
                )
            else:
                return Waveform(
                    np.concatenate([self._ndarray, other.get_ndarray()])
                )
        else:
            raise RuntimeError

    def broadcast(self) -> None:
        broadcast(self._ndarray, fs=qkc().fs)

    def bounce(self, file_path: String = "waveform.wav") -> None:
        bounce(self._ndarray, file_path=file_path, fs=qkc().fs)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other, r=False)

    def __rmatmul__(self, other):
        return self.matmul(other, r=True)

    def __len__(self):
        return len(self._ndarray)


class Stream(Waveform):
    pass


class Controller(Waveform):
    pass
