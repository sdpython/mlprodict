# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Pow(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        return (numpy.power(a, b).astype(a.dtype), )

    def _infer_shapes(self, x, b):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        """
        return (x, )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'power')
