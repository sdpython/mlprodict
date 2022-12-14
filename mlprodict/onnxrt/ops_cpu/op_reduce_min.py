# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunReduceNumpy, OpRun


class ReduceMin(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceMin.atts,
                                  **options)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        axes = tuple(self.axes) if self.axes else None
        return (numpy.minimum.reduce(data, axis=axes,  # pylint: disable=E1123
                                     keepdims=self.keepdims == 1), )


class ReduceMin_18(OpRun):

    atts = {'keepdims': 1, 'noop_with_empty_axes': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ReduceMin_18.atts,
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
            return (numpy.minimum.reduce(
                data, axis=axes if axes else None,
                keepdims=self.keepdims,
                dtype=data.dtype), )
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r}.") from e


if onnx_opset_version() >= 18:
    ReduceMin = ReduceMin_18
else:  # pragma: no cover
    ReduceMin = ReduceMin_13
