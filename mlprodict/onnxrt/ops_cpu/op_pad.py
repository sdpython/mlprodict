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


def onnx_pad(data, pads, constant_value=None, mode='constant'):
    """
    Implements :epkg:`numpy:pad` based on ONNX signature.

    :param data: data to pad
    :param pads: tensor of integers indicating the number of
        padding elements to add or remove (if negative) at the
        beginning and end of each axis. For 2D input tensor, it
        is the number of pixels. `pads` should be a 1D tensor of
        shape `[2 * input_rank]`. `pads` format should be:
        `[x1_begin, x2_begin,...,x1_end, x2_end,...]`, where `xi_begin` is
        the number of pad values added at the beginning of axis `i`
        and xi_end, the number of pad values added at the end of axis `i`.
    :param constant_value: A scalar value to be used if the mode chosen is
        `constant` (by default it is 0, empty string or False).
    :param mode: Supported modes: `constant`(default), `reflect`, `edge`
    :return: tensor after padding
    """
    return _pad_impl(
        data, pads, mode=mode,
        constant_values=constant_value or numpy.array(
            [0], dtype=data.dtype.type))


class Pad(OpRun):

    atts = {'mode': b'constant'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Pad.atts,
                       **options)
        self.mode_ = self.mode.decode('ascii')

    def _run(self, data, pads, constant_value=None):  # pylint: disable=W0221
        if constant_value is None:
            constant_value = 0
        return (_pad_impl(data, pads, mode=self.mode_,
                          constant_values=constant_value), )

    def _infer_shapes(self, data, pads, constant_value=None):  # pylint: disable=E0202,W0221
        return (ShapeObject(None, data.dtype), )

    def _infer_types(self, data, pads, constant_value=None):  # pylint: disable=E0202,W0221
        return (data, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res
