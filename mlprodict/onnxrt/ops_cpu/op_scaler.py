# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRunUnaryNum


class Scaler(OpRunUnaryNum):

    atts = {'offset': None, 'scale': None}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Scaler.atts,
                               **options)

    def _run(self, x):  # pylint: disable=W0221

        if self.inplaces.get(0, False):
            x -= self.offset
            x *= self.scale
            return (x, )
        else:
            return ((x - self.offset) * self.scale, )
