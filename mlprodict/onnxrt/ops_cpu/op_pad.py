# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


def _pad_impl(data, raw_pads, mode, constant_values=0.0):
    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise RuntimeError(  # pragma: no cover
            'The number of elements in raw_pads should be 2 * data_rank')

    half = raw_pads.shape[0] // 2
    pad_width = tuple((raw_pads[i], raw_pads[i + half])
                      for i in range(0, half))

    if mode == 'constant':
        return numpy.pad(data, pad_width=pad_width, mode=mode,
                         constant_values=constant_values)
    return numpy.pad(data, pad_width=pad_width, mode=mode)


class Pad(OpRun):

    atts = {'mode': b'constant'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Pad.atts,
                       **options)
        self.mode_ = self.mode.decode('ascii')

    def _run(self, data, pads, constant_value=None):  # pylint: disable=W0221
        return (_pad_impl(data, pads, mode=self.mode_,
                          constant_values=constant_value), )

    def _infer_shapes(self, data, pads, constant_value=None):  # pylint: disable=E0202,W0221
        """
        Returns an empty shape by default.
        """
        return (ShapeObject(None, data.dtype), )
