# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import itertools
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


def _pool_get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                        strides_spatial, output_spatial_shape):
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in (b'SAME_UPPER', b'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                kernel_spatial_shape[i] - input_spatial_shape[i]
    elif auto_pad == b'VALID':
        pass
    return pad_shape


def _pool_get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                           strides_spatial):
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in (b'SAME_UPPER', b'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
            out_shape[i] = int(
                numpy.ceil(
                    float(input_spatial_shape[i]) / float(strides_spatial[i])))
    elif auto_pad == b'VALID':
        for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
            out_shape[i] = int(
                numpy.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) /
                           float(strides_spatial[i])))
    return out_shape


def _pool_impl(padded, x_shape, kernel_shape, strides_shape,
               out_shape, pad_shape, pooling_type,
               count_include_pad=0):
    spatial_size = len(x_shape) - 2
    y = numpy.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(
            range(x_shape[0]), range(x_shape[1]),
            *[range(int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) /
                        strides_shape[i] + 1))
              for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        window_vals = numpy.array(
            [window[i] for i in list(
                itertools.product(
                    *[range(strides_shape[i] * shape[i + 2],
                            strides_shape[i] * shape[i + 2] + kernel_shape[i])
                      for i in range(spatial_size)]))])
        if pooling_type == b'AVG':
            f = numpy.average
        elif pooling_type == b'MAX':
            f = numpy.max
        else:
            raise NotImplementedError(  # pragma: no cover
                "Pooling type '{}' does not support. Should be AVG, MAX."
                "".format(pooling_type))

        if count_include_pad == 1 and pooling_type == b'AVG':
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[numpy.where(~numpy.isnan(window_vals))])
    return y.astype(numpy.float32)


class MaxPool(OpRun):

    atts = {'auto_pad': b'NOTSET', 'ceil_mode': 0, 'dilations': [],
            'kernel_shape': [], 'pads': [], 'storage_order': 0,
            'strides': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=MaxPool.atts,
                       **options)
        self.auto_pad_ = self.auto_pad.decode('ascii')

    def _run(self, x):  # pylint: disable=W0221
        if self.pads is None:
            pads = [1 for d in x.shape]
        else:
            pads = self.pads
        raise NotImplementedError()

    def _infer_shapes(self, x):  # pylint: disable=E0202,W0221
        """
        Returns an empty shape by default.
        """
        return (ShapeObject(None, x.dtype), )
