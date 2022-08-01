# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import math
import numpy
from ._op import OpRun


class LRN(OpRun):

    atts = {
        'alpha': 9.999999747378752e-05,
        'beta': 0.75,
        'bias': 1.,
        'size': 3,
    }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=LRN.atts,
                       **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(x.shape) != 4:
            raise RuntimeError(  # pragma: no cover
                f"LRN only applies on 4D tensors but shape is {x.shape!r}.")
        square_sum = numpy.zeros(x.shape).astype(x.dtype)
        for ind in numpy.ndindex(x.shape):
            n, c, h, w = ind
            begin = max(0, c - int(math.floor((self.size - 1) / 2)))
            end = min(5, c + int(math.ceil((self.size - 1) / 2)) + 1)
            square_sum[n, c, h, w] = numpy.sum(x[n, begin:end, h, w] ** 2)
        y = x / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        return (y.astype(x.dtype), )
