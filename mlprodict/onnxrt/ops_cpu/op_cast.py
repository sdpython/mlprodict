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
            self._dtype = numpy.float32
        elif self.to == TensorProto.DOUBLE:  # pylint: disable=E1101
            self._dtype = numpy.float64
        elif self.to == TensorProto.UINT8:  # pylint: disable=E1101
            self._dtype = numpy.uint8
        elif self.to == TensorProto.INT8:  # pylint: disable=E1101
            self._dtype = numpy.int8
        elif self.to == TensorProto.INT16:  # pylint: disable=E1101
            self._dtype = numpy.int16
        elif self.to == TensorProto.INT32:  # pylint: disable=E1101
            self._dtype = numpy.int32
        elif self.to == TensorProto.INT64:  # pylint: disable=E1101
            self._dtype = numpy.int64
        elif self.to == TensorProto.UINT16:  # pylint: disable=E1101
            self._dtype = numpy.uint16
        elif self.to == TensorProto.UINT32:  # pylint: disable=E1101
            self._dtype = numpy.uint32
        elif self.to == TensorProto.UINT64:  # pylint: disable=E1101
            self._dtype = numpy.uint64
        elif self.to == TensorProto.BOOL:  # pylint: disable=E1101
            self._dtype = numpy.bool_
        elif self.to == TensorProto.STRING:  # pylint: disable=E1101
            self._dtype = numpy.str_
        elif self.to == TensorProto.FLOAT16:  # pylint: disable=E1101
            self._dtype = numpy.float16
        elif self.to == TensorProto.COMPLEX64:  # pylint: disable=E1101
            self._dtype = numpy.complex64
        elif self.to == TensorProto.COMPLEX128:  # pylint: disable=E1101
            self._dtype = numpy.complex128
        else:
            raise ValueError(  # pragma: no cover
                "Unexpected value for to='{}'.".format(
                    self.to))  # pylint: disable=E1101
        self._cast = lambda x: x.astype(self._dtype)

    def _run(self, x):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(x)
        return (self._cast(x), )

    def _run_inplace(self, x):
        if x.dtype == self._dtype:
            return (x, )
        return (self._cast(x), )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (x.copy(dtype=self._dtype), )

    def _infer_types(self, x):  # pylint: disable=W0221
        return (self._dtype, )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res
