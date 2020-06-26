# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum, RuntimeTypeError


class Imputer(OpRunUnaryNum):

    atts = {'imputed_value_floats': numpy.empty(0, dtype=numpy.float32),
            'imputed_value_int64s': numpy.empty(0, dtype=numpy.int64),
            'replaced_value_float': 0.,
            'replaced_value_int64': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Imputer.atts,
                               **options)
        if len(self.imputed_value_floats) > 0:
            self.values = self.imputed_value_floats
            self.replace = self.replaced_value_float
        elif len(self.imputed_value_int64s) > 0:
            self.values = self.imputed_value_int64s
            self.replace = self.replaced_value_int64
        else:
            raise ValueError("Missing are not defined.")  # pragma: no cover

    def _run(self, x):  # pylint: disable=W0221
        if len(x.shape) != 2:
            raise RuntimeTypeError(
                "x must be a matrix but shape is {}".format(x.shape))
        if self.values.shape[0] not in (x.shape[1], 1):
            raise RuntimeTypeError(  # pragma: no cover
                "Dimension mismatch {} != {}".format(
                    self.values.shape[0], x.shape[1]))
        x = x.copy()
        if numpy.isnan(self.replace):
            for i in range(0, x.shape[1]):
                val = self.values[min(i, self.values.shape[0] - 1)]
                x[numpy.isnan(x[:, i]), i] = val
        else:
            for i in range(0, x.shape[1]):
                val = self.values[min(i, self.values.shape[0] - 1)]
                x[x[:, i] == self.replace, i] = val

        return (x, )
