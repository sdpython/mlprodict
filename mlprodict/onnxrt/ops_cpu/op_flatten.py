# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnary


class Flatten(OpRunUnary):

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=Flatten.atts,
                            **options)

    def _run(self, x):  # pylint: disable=W0221
        i = self.axis
        shape = x.shape
        new_shape = ((1, -1) if i == 0 else
                     (numpy.prod(shape[0:i]).astype(int), -1))
        return (x.reshape(new_shape), )
