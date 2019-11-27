# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunArg


def _argmax_use_numpy_select_last_index(
        data, axis=0, keepdims=True):
    data = numpy.flip(data, axis)
    result = numpy.argmax(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


class _ArgMax(OpRunArg):

    def __init__(self, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRunArg.__init__(self, onnx_node, desc=desc,
                          expected_attributes=expected_attributes,
                          **options)

    def _run(self, data):  # pylint: disable=W0221
        r = numpy.argmax(data, axis=self.axis)
        if self.keepdims == 0:
            r = r.astype(numpy.int64)
        else:
            if len(data.shape) == 2:
                if len(r.shape) == 2:
                    r = r.astype(numpy.int64)
                else:
                    if self.axis == 0:
                        r = r.astype(numpy.int64)[numpy.newaxis, :]
                    else:
                        r = r.astype(numpy.int64)[:, numpy.newaxis]
            else:
                raise NotImplementedError(
                    "keepdims not implemented for dimension > 2.")
        return (r, )


class ArgMax_11(_ArgMax):

    atts = {'axis': 0, 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        _ArgMax.__init__(self, onnx_node, desc=desc,
                         expected_attributes=ArgMax_11.atts,
                         **options)


class ArgMax_12(_ArgMax):

    atts = {'axis': 0, 'keepdims': 1, 'select_last_index': 0}

    def __init__(self, onnx_node, desc=None, **options):
        _ArgMax.__init__(self, onnx_node, desc=desc,
                         expected_attributes=ArgMax_12.atts,
                         **options)

    def _run(self, data):  # pylint: disable=W0221
        if self.select_last_index == 0:
            return _ArgMax._run(self, data)
        return (_argmax_use_numpy_select_last_index(
            data, axis=self.axis, keepdims=self.keepdims), )


if onnx_opset_version() >= 12:
    ArgMax = ArgMax_12
else:
    ArgMax = ArgMax_11
