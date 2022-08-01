# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from .op_conv_ import ConvFloat, ConvDouble  # pylint: disable=E0611,E0401


class Conv(OpRun):

    atts = {'auto_pad': 'NOTSET', 'group': 1,
            'dilations': [1, 1],
            'kernel_shape': [],
            'pads': [],
            'strides': [1, 1]}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Conv.atts,
                       **options)
        self._init()

    def _init(self):
        self.rt32_ = ConvFloat()
        self.rt64_ = ConvDouble()
        for rt in [self.rt32_, self.rt64_]:
            rt.init(self.auto_pad,
                    numpy.array(self.dilations, dtype=numpy.int64),
                    self.group,
                    numpy.array(self.kernel_shape, dtype=numpy.int64),
                    numpy.array(self.pads, dtype=numpy.int64),
                    numpy.array(self.strides, dtype=numpy.int64))

    def _run(self, X, W, B=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if X is None:
            raise ValueError(  # pragma: no cover
                "X cannot be None for operator %r, ONNX=%r" % (
                    type(self), self.onnx_node))
        if min(X.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. X.shape={X.shape!r}.")
        if min(W.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. W.shape={W.shape!r}.")
        if B is not None and min(B.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. B.shape={B.shape!r}.")
        if X.dtype == numpy.float32:
            return (self.rt32_.compute(X, W, B), )
        return (self.rt64_.compute(X, W, B), )
