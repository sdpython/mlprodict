# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from .op_conv_transpose_ import (  # pylint: disable=E0611,E0401
    ConvTransposeFloat, ConvTransposeDouble)


class ConvTranspose(OpRun):

    atts = {'auto_pad': 'NOTSET', 'group': 1,
            'dilations': [],
            'kernel_shape': [],
            'pads': [],
            'strides': [],
            'output_padding': [],
            'output_shape': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ConvTranspose.atts,
                       **options)
        self._init()

    def _init(self):
        self.rt32_ = ConvTransposeFloat()
        self.rt64_ = ConvTransposeDouble()
        for rt in [self.rt32_, self.rt64_]:
            rt.init(self.auto_pad,
                    numpy.array(self.dilations, dtype=numpy.int64),
                    self.group,
                    numpy.array(self.kernel_shape, dtype=numpy.int64),
                    numpy.array(self.pads, dtype=numpy.int64),
                    numpy.array(self.strides, dtype=numpy.int64),
                    numpy.array(self.output_padding, dtype=numpy.int64),
                    numpy.array(self.output_shape, dtype=numpy.int64))

    def _run(self, X, W, B=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if X.dtype == numpy.float32:
            return (self.rt32_.compute(X, W, B), )
        return (self.rt64_.compute(X, W, B), )
