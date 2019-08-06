# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Clip(OpRunUnaryNum):

    atts = {'min': -3.4028234663852886e+38,
            'max': 3.4028234663852886e+38}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Clip.atts,
                               **options)

    def _run(self, data):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            res = numpy.clip(data, self.min, self.max, out=data)
        else:
            res = numpy.clip(data, self.min, self.max)
        return (res, ) if res.dtype == data.dtype else (res.astype(data.dtype), )
