# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.onnx_pb import TensorProto
from ._op import OpRun


class Cast(OpRun):

    atts = {'to': None}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Cast.atts,
                       **options)
        # type help(TensorProto) to see all the possible values
        if self.to == TensorProto.FLOAT:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.float32)
        elif self.to == TensorProto.INT32:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.int32)
        elif self.to == TensorProto.INT64:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.int64)
        elif self.to == TensorProto.BOOL:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.bool)
        else:
            raise ValueError("Unexpected value for to='{}'.".format(
                self.to))  # pylint: disable=E1101

    def _run(self, x):  # pylint: disable=W0221
        return (self._cast(x), )
