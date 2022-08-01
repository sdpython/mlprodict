# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.onnx_pb import TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op import OpRun


class Cast(OpRun):

    atts = {'to': None}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Cast.atts,
                       **options)
        if self.to == TensorProto.STRING:  # pylint: disable=E1101
            self._dtype = numpy.str_
        else:
            self._dtype = TENSOR_TYPE_TO_NP_TYPE[self.to]
        self._cast = lambda x: x.astype(self._dtype)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return self._run_inplace(x)
        return (self._cast(x), )

    def _run_inplace(self, x):
        if x.dtype == self._dtype:
            return (x, )
        return (self._cast(x), )


class CastLike(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, x, y, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return self._run_inplace(x, y)
        return (x.astype(y.dtype), )

    def _run_inplace(self, x, y):
        if x.dtype == y.dtype:
            return (x, )
        return (x.astype(y.dtype), )
