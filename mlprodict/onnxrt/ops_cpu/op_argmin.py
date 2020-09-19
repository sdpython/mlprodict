# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunArg


def _argmin(data, axis=0, keepdims=True):
    result = numpy.argmin(data, axis=axis)
    if keepdims and len(result.shape) < len(data.shape):
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


def _argmin_use_numpy_select_last_index(
        data, axis=0, keepdims=True):
    data = numpy.flip(data, axis)
    result = numpy.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


class _ArgMin(OpRunArg):
    """
    Base class for runtime for operator `ArgMin
    <https://github.com/onnx/onnx/blob/master/docs/
    Operators.md#ArgMin>`_.
    """

    def __init__(self, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRunArg.__init__(self, onnx_node, desc=desc,
                          expected_attributes=expected_attributes,
                          **options)

    def _run(self, data):  # pylint: disable=W0221
        return (_argmin(data, axis=self.axis, keepdims=self.keepdims), )


class ArgMin_11(_ArgMin):

    atts = {'axis': 0, 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        _ArgMin.__init__(self, onnx_node, desc=desc,
                         expected_attributes=ArgMin_11.atts,
                         **options)

    def to_python(self, inputs):
        return ('import numpy\nfrom mlprodict.onnxrt.ops_cpu.op_argmin import _argmin',
                'return _argmin(%s, axis=axis, keepdims=keepdims)' % inputs[0])


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

    def to_python(self, inputs):
        lines = [
            "if select_last_index == 0:",
            "    return _argmin({0}, axis=axis, keepdims=keepdims)",
            "return _argmin_use_numpy_select_last_index(",
            "    {0}, axis=axis, keepdims=keepdims)"]
        return ('import numpy\nfrom mlprodict.onnxrt.ops_cpu.op_argmin import _argmin, _argmin_use_numpy_select_last_index',
                "\n".join(lines).format(inputs[0]))


if onnx_opset_version() >= 12:
    ArgMin = ArgMin_12
else:
    ArgMin = ArgMin_11  # pragma: no cover
