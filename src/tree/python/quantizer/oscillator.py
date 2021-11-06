from quantizer.kernel.kontext import kntxt
from quantizer.kernel.util import (
    Array, Scalar, String, broadcast, bounce, transpose
)
from quantizer.kernel.waveform import Controller
from quantizer.controller import (
    BipolarController, StaticController, cast
)
from quantizer.wavetable import Wavetable, Sine
import numpy as np


class Oscillator(BipolarController):
    def __init__(
        self,
        wt: Wavetable = Sine(),
        dt: (Scalar, Array, Controller) = None,
        f: (Scalar, Array, Controller) = None,
        p: (Scalar, Array, Controller) = None,
        a: (Scalar, Array, Controller) = None,
        detune: (Scalar, Array, Controller) = 0,
    ):
        super().__init__()
        if not dt:
            dt = StaticController(1 / kntxt().fs)
        if not f:
            f = kntxt().f
        if not p:
            p = kntxt().p
        if not a:
            a = kntxt().a
        wt = wt
        dt = cast(dt)
        f = cast(f)
        p = cast(p)
        a = cast(a)
        detune = cast(detune)
        fd = self.detune(f, detune)
        phase_derivative = (
            2 * np.pi
            * fd.get_ndarray()
            * dt.get_ndarray()
        )
        phase_integral = np.cumsum(phase_derivative)
        phase = phase_integral + p.get_ndarray()
        self._ndarray = a.get_ndarray() * wt.render(phase)

    @staticmethod
    def detune(f: Controller, detune: Controller) -> Controller:
        return Controller(transpose(f.get_ndarray(), ct=detune.get_ndarray()))

    def broadcast(self) -> None:
        broadcast(self._ndarray, fs=kntxt().fs)

    def bounce(self, file_path: String = "oscillator.wav") -> None:
        bounce(self._ndarray, file_path=file_path, fs=kntxt().fs)


Osc = Oscillator
