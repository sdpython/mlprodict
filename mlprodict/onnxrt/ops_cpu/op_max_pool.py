# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import itertools
import numpy
from ..shape_object import ShapeObjectFct
from ._op import OpRun
from .op_max_pool_ import MaxPoolFloat, MaxPoolDouble  # pylint: disable=E0611,E0401


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
        self.nb_outputs = len(onnx_node.output)
        self._init()

    def _init(self):
        self.rt32_ = MaxPoolFloat()
        self.rt64_ = MaxPoolDouble()
        for rt in [self.rt32_, self.rt64_]:
            rt.init(self.auto_pad,
                    numpy.array(self.dilations, dtype=numpy.int64),
                    self.ceil_mode,
                    self.storage_order,
                    numpy.array(self.kernel_shape, dtype=numpy.int64),
                    numpy.array(self.pads, dtype=numpy.int64),
                    numpy.array(self.strides, dtype=numpy.int64))

    def _run(self, X):  # pylint: disable=W0221
        if X.dtype == numpy.float32:
            res = self.rt32_.compute(X)
        else:
            res = self.rt64_.compute(X)
        if self.nb_outputs == 1:
            return res[:1]
        return res

    def _infer_shapes(self, X):  # pylint: disable=W0221

        def compute_shape1(xshape):
            xs = numpy.ones(xshape, dtype=numpy.float32)
            res, _ = self.rt32_.compute(xs)
            return res.shape

        def compute_shape2(xshape):
            xs = numpy.ones(xshape, dtype=numpy.float32)
            _, res2 = self.rt32_.compute(xs)
            return res2.shape

        if self.nb_outputs == 1:
            return (ShapeObjectFct(compute_shape1, X, name="MaxPool", dtype=X.dtype), )
        return (ShapeObjectFct(compute_shape1, X, name="MaxPool", dtype=X.dtype),
                ShapeObjectFct(compute_shape2, X, name="MaxPool", dtype=X.dtype))

    def _infer_types(self, X):  # pylint: disable=W0221
        if self.nb_outputs == 1:
            return (X, )
        return (X, X)

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res
