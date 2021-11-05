from quantizer.kernel.util import Array
from quantizer.kernel.waveform import Stream


class MonoStream(Stream):
    def __init__(self, ndarray: Array):
        assert ndarray.ndim == 1
        super().__init__(ndarray)


class StereoStream(Stream):
    def __init__(self, ndarray: Array):
        assert ndarray.ndim == 2
        super().__init__(ndarray)


class MultiStream(Stream):
    def __init__(self, ndarray: Array):
        assert ndarray.ndim > 2
        super().__init__(ndarray)
