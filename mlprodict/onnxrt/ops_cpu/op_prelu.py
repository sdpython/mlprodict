# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class PRelu(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, x, slope, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.where(x > 0, x, x * slope), )

    def to_python(self, inputs):
        return ('import numpy',
                "return numpy.where({0} > 0, {0}, {0} * {1})".format(
                    inputs[0], inputs[1]))
