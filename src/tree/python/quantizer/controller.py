from quantizer.kernel.kontext import kntxt, Controller
from quantizer.kernel.util import Array, Integer, Scalar
import numpy as np


class UnipolarController(Controller):
    pass


class BipolarController(Controller):
    pass


class StaticController(Controller):
    def __init__(
        self,
        scalar: Scalar,
        nsamples: Integer = None,
    ):
        if not nsamples:
            nsamples = len(kntxt())
        ndarray = np.zeros(nsamples)
        ndarray.fill(scalar)
        super().__init__(ndarray)


def cast(arg: (Scalar, Array, Controller)) -> Controller:
    if isinstance(arg, Scalar):
        return StaticController(arg)
    elif isinstance(arg, Array):
        return Controller(arg)
    else:
        try:
            return Controller(arg.get_ndarray())
        except AttributeError:
            raise RuntimeError
