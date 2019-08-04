# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Transpose(OpRunUnaryNum):

    atts = {'perm': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Transpose.atts,
                               **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.transpose(data, axes=self.perm), )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (x.transpose(perm=self.perm), )
