# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum
from ._op_numpy_helper import numpy_dot_inplace


class LinearRegressor(OpRunUnaryNum):

    atts = {'coefficients': None, 'intercepts': None,
            'targets': 1, 'post_transform': b'NONE'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=LinearRegressor.atts,
                               **options)
        if not isinstance(self.coefficients, numpy.ndarray):
            raise TypeError(  # pragma: no cover
                "coefficient must be an array not {}.".format(
                    type(self.coefficients)))
        n = self.coefficients.shape[0] // self.targets
        self.coefficients = self.coefficients.reshape(self.targets, n).T

    def _run(self, x):  # pylint: disable=W0221
        score = numpy_dot_inplace(self.inplaces, x, self.coefficients)
        if self.intercepts is not None:
            score += self.intercepts
        if self.post_transform == b'NONE':
            pass
        else:
            raise NotImplementedError(  # pragma: no cover
                "Unknown post_transform: '{}'.".format(
                    self.post_transform))
        return (score, )
