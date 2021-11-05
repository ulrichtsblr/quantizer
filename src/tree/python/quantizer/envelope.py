from quantizer.kernel.kontext import kntxt
from quantizer.kernel.util import Array, Callable, Dict
from quantizer.controller import UnipolarController
from quantizer.sequencer import Event, Pattern
from numba import jit
import numpy as np


class Contour:
    def __init__(
        self,
        head: Dict,
        sustain: Callable,
        tail: Dict,
    ):
        self.head_a = head['a']
        self.head_t = head['t']
        self.head_exp = head['exp']
        self.sustain = sustain
        self.tail_a = tail['a']
        self.tail_t = tail['t']
        self.tail_exp = tail['exp']

    def render(self, e: Event) -> Array:
        nsamples = e.get_nsamples()
        h = self._segment(self.head_t, self.head_a, kntxt().fs)
        if nsamples <= len(h):
            hs = h[:nsamples]
        else:
            s = np.zeros(nsamples - len(h))
            s.fill(self.sustain(e))
            hs = np.concatenate([h, s])
        t = self._segment(self.tail_t, self.tail_a, kntxt().fs)
        hst = np.concatenate([hs, t])
        return hst

    @staticmethod
    @jit(nopython=True, cache=True)
    def _segment(t, a, fs):
        y = np.linspace(
            a[0],
            a[1],
            round(t[0] * fs),
        )
        for i in range(1, len(t)):
            y = np.concatenate((
                y,
                np.linspace(
                    a[i],
                    a[i + 1],
                    round(t[i] * fs),
                )
            ))
        return y


class Envelope(UnipolarController):
    def __init__(self, pattern, contour, reactive=True, lazy=False):
        super().__init__()
        self.pattern = pattern
        self.contour = contour
        self.reactive = reactive
        self.lazy = lazy
        self._ndarray = np.zeros(len(kntxt()))
        val = 0
        idx = 0
        for e in self.pattern.get_events():
            nsamples = e.get_nsamples()
            if not e.get_idle():
                c = self.contour.render(e)
                self._ndarray[idx: idx + len(c)] = c
                val = c[-1]
            else:
                if not reactive:
                    self._ndarray[idx: idx + nsamples] = val
            idx += nsamples


class FMEnvelope(Envelope):

    def __init__(self, pattern: Pattern):
        super().__init__(
            pattern,
            Contour(
                head={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
                sustain=lambda e: e.get_f(),
                tail={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
            ),
            reactive=False,
        )


class PMEnvelope(Envelope):

    def __init__(self, pattern: Pattern):
        super().__init__(
            pattern,
            Contour(
                head={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
                sustain=lambda e: e.get_p(),
                tail={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
            ),
            reactive=False,
        )


class AMEnvelope(Envelope):

    def __init__(self, pattern: Pattern):
        super().__init__(
            pattern,
            Contour(
                head={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
                sustain=lambda e: e.get_a(),
                tail={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
            ),
            reactive=False,
        )


class DeltaEnvelope(Envelope):
    def __init__(
            self,
            pattern: Pattern,
            a=10e-3,
            d=100e-3,
    ):
        super().__init__(
            pattern,
            Contour(
                head={
                    'a': [0., 1., 0.],
                    't': [float(a), float(d)],
                    'exp': [1., 1.],
                },
                sustain=lambda e: 0.,
                tail={
                    'a': [0., 0.],
                    't': [0.],
                    'exp': [1.],
                },
            ),
        )


class ADSREnvelope(Envelope):
    def __init__(
            self,
            pattern: Pattern,
            a=10e-3,
            d=100e-3,
            s=0.5,
            r=1000e-3,
            lazy=False,
    ):
        super().__init__(
            pattern,
            Contour(
                head={
                    'a': [0., 1., float(s)],
                    't': [float(a), float(d)],
                    'exp': [1., 1.],
                },
                sustain=lambda e: float(s),
                tail={
                    'a': [float(s), 0.],
                    't': [float(r)],
                    'exp': [1.],
                },
            ),
            lazy=lazy,
        )


FME = FMEnvelope
PME = PMEnvelope
AME = AMEnvelope
ADSRE = ADSREnvelope
DE = DeltaEnvelope
