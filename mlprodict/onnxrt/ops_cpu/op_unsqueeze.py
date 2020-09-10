# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Unsqueeze(OpRunUnaryNum):

    atts = {'axes': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Unsqueeze.atts,
                               **options)
        if isinstance(self.axes, numpy.ndarray):
            self.axes = tuple(self.axes)
        elif self.axes in [[], tuple()]:
            self.axes = None
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)
        self.axes = tuple(sorted(self.axes))

    def _run(self, data):  # pylint: disable=W0221
        sq = data
        for ax in self.axes:
            sq = numpy.expand_dims(sq, axis=ax)
        return (sq, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (x.unsqueeze(axes=self.axes), )
