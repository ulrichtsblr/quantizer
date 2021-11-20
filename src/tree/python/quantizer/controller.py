from quantizer.kernel.context import Controller


class UnipolarController(Controller):
    pass


class BipolarController(Controller):
    pass


class StaticController(Controller):
    def __init__(self, scalar):
        """
        :param scalar: scalar
        :type scalar: Scalar
        :rtype None
        """
        super().__init__(
            Controller.cast(scalar).get_ndarray()
        )
