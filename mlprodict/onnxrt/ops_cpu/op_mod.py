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

    def _run(self, a, b):  # pylint: disable=W0221
        return (numpy.nan_to_num(numpy.mod(a, b)), )

    def _infer_shapes(self, x, b):  # pylint: disable=W0221
        return (x, )

    def _infer_types(self, x, b):  # pylint: disable=W0221
        return (x, )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'mod')
