# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunArg


def _argmin_use_numpy_select_last_index(
        data, axis=0, keepdims=True):
    data = numpy.flip(data, axis)
    result = numpy.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


class _ArgMin(OpRunArg):

    def __init__(self, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRunArg.__init__(self, onnx_node, desc=desc,
                          expected_attributes=expected_attributes,
                          **options)

    def _run(self, data):  # pylint: disable=W0221
        r = numpy.argmin(data, axis=self.axis)
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


class ArgMin_11(_ArgMin):

    atts = {'axis': 0, 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        _ArgMin.__init__(self, onnx_node, desc=desc,
                         expected_attributes=ArgMin_11.atts,
                         **options)


class ArgMin_12(_ArgMin):

    atts = {'axis': 0, 'keepdims': 1, 'select_last_index': 0}

    def __init__(self, onnx_node, desc=None, **options):
        _ArgMin.__init__(self, onnx_node, desc=desc,
                         expected_attributes=ArgMin_12.atts,
                         **options)

    def _run(self, data):  # pylint: disable=W0221
        if self.select_last_index == 0:
            return _ArgMin._run(self, data)
        return (_argmin_use_numpy_select_last_index(
            data, axis=self.axis, keepdims=self.keepdims), )


if onnx_opset_version() >= 12:
    ArgMin = ArgMin_12
else:
    ArgMin = ArgMin_11
