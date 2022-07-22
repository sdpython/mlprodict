# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Mod(OpRun):

    atts = {'fmod': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Mod.atts,
                       **options)

    def _run(self, a, b, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if a.dtype in (numpy.float16, numpy.float32, numpy.float64):
            return (numpy.nan_to_num(numpy.fmod(a, b)), )
        return (numpy.nan_to_num(numpy.mod(a, b)), )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'mod')
