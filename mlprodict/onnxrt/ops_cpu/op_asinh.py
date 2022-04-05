# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Asinh(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, x):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return self._run_inplace(x)
        return (numpy.arcsinh(x), )

    def _run_inplace(self, x):
        return (numpy.arcsinh(x, out=x), )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'arcsinh')
