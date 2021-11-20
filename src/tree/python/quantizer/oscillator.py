from quantizer.kernel.util import Array, Scalar, qkc
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
    ):
        super().__init__()
        if not dt:
            dt = StaticController(1 / qkc().fs)
        if not f:
            f = qkc().f
        if not p:
            p = qkc().p
        if not a:
            a = qkc().a
        wt = wt
        dt = cast(dt)
        f = cast(f)
        p = cast(p)
        a = cast(a)
        phase_derivative = (
            2 * np.pi
            * f.get_ndarray()
            * dt.get_ndarray()
        )
        phase_integral = np.cumsum(phase_derivative)
        phase = phase_integral + p.get_ndarray()
        self._ndarray = a.get_ndarray() * wt.render(phase)


Osc = Oscillator
