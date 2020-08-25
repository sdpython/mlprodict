# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinary


class And(OpRunBinary):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinary.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        return (numpy.logical_and(a, b), )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, "logical_and")
