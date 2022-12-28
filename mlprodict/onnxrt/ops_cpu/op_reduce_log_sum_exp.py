# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunReduceNumpy, OpRun


class ReduceLogSumExp_1(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceLogSumExp_1.atts,
                                  **options)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        tax = tuple(self.axes) if self.axes else None
        data_max = data.copy()
        ind = numpy.isinf(data_max)
        data_max[ind] = -numpy.inf
        mx = data_max.max(axis=tax, keepdims=True)
        sub = numpy.subtract(data, mx)
        exp = numpy.exp(sub, out=sub)
        mxs = numpy.sum(exp, axis=tax,
                        keepdims=True,
                        dtype=data.dtype)
        res = numpy.log(mxs) + mx
        if not self.keepdims:
            res = numpy.squeeze(res, axis=tax)
        return (res, )


class ReduceLogSumExp_18(OpRun):

    atts = {'keepdims': 1, 'noop_with_empty_axes': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ReduceLogSumExp_18.atts,
                       **options)

    def _run(self, data, axes=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if ((axes is None or len(axes.shape) == 0 or axes.shape[0] == 0) and
                self.noop_with_empty_axes):
            return (data, )
        if ((axes is not None and len(axes.shape) > 0 and axes.shape[0] > 0) and
                not isinstance(axes, int)):
            if isinstance(axes, numpy.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes.ravel().tolist()) if len(axes) > 0 else None
        try:
            tax = tuple(axes) if axes else None
            data_max = data.copy()
            ind = numpy.isinf(data_max)
            data_max[ind] = -numpy.inf
            mx = data_max.max(axis=tax, keepdims=True)
            sub = numpy.subtract(data, mx)
            exp = numpy.exp(sub, out=sub)
            mxs = numpy.sum(exp, axis=tax,
                            keepdims=True,
                            dtype=data.dtype)
            res = numpy.log(mxs) + mx
            if not self.keepdims:
                res = numpy.squeeze(res, axis=tax)
            return (res, )
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r}.") from e


if onnx_opset_version() >= 18:
    ReduceLogSumExp = ReduceLogSumExp_18
else:  # pragma: no cover
    ReduceLogSumExp = ReduceLogSumExp_1
