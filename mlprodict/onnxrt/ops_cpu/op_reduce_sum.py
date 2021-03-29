# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunReduceNumpy, RuntimeTypeError, OpRun


class ReduceSum_1(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceSum_1.atts,
                                  **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.sum(data, axis=self.axes,
                          keepdims=self.keepdims,
                          dtype=data.dtype), )


class ReduceSum_11(ReduceSum_1):

    def __init__(self, onnx_node, desc=None, **options):
        ReduceSum_1.__init__(self, onnx_node, desc=desc, **options)


class ReduceSum_13(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1, 'noop_with_empty_axes': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceSum_13.atts,
                                  **options)

    def run(self, data, axes=None):  # pylint: disable=E0202,W0221
        """
        Calls method ``_run``.
        """
        res = self._run(data, axes=axes)
        if not self.keepdims and not isinstance(res[0], numpy.ndarray):
            res = (numpy.array([res[0]], dtype=res[0].dtype), )
        if res[0].dtype != data.dtype:
            raise RuntimeTypeError(  # pragma: no cover
                "Output type mismatch: input '{}' != output '{}' "
                "(operator '{}')".format(
                    data.dtype, res[0].dtype, self.__class__.__name__))
        return res

    def _run_no_checks_(self, x, axes=None):  # pylint: disable=W0221
        return OpRun.run(self, x, axes)

    def _run(self, data, axes=None):  # pylint: disable=W0221
        if axes is None and self.noop_with_empty_axes:
            return (data, )
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        try:
            return (numpy.sum(data, axis=axes,
                              keepdims=self.keepdims,
                              dtype=data.dtype), )
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                "Unable to reduce shape %r with axes=%r." % (
                    data.shape, axes)) from e

    def infer_shapes(self, data, axes=None):  # pylint: disable=E0202,W0221
        return self._infer_shapes(data, axes=axes)

    def _infer_shapes(self, data, axes=None):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        """
        sh = data.reduce(axes, self.keepdims,  # pylint: disable=E1101
                         dtype=numpy.int64)  # pylint: disable=E1101
        return (sh, )


if onnx_opset_version() >= 13:
    ReduceSum = ReduceSum_13
elif onnx_opset_version() >= 11:  # pragma: no cover
    ReduceSum = ReduceSum_11
else:  # pragma: no cover
    ReduceSum = ReduceSum_1
