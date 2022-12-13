# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun, OpRunReduceNumpy


class ReduceMean_13(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceMean_13.atts,
                                  **options)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.mean(data, axis=self.axes,
                           keepdims=self.keepdims,
                           dtype=data.dtype), )


class ReduceMean_18(OpRun):

    atts = {'keepdims': 1, 'noop_with_empty_axes': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ReduceMean_18.atts,
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
            return (numpy.mean(data, axis=axes if axes else None,
                               keepdims=self.keepdims,
                               dtype=data.dtype), )
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r}.") from e


if onnx_opset_version() >= 18:
    ReduceMean = ReduceMean_18
else:  # pragma: no cover
    ReduceMean = ReduceMean_13
