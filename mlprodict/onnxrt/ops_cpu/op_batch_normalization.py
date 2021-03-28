# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun


def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    y = s * (x - mean) / numpy.sqrt(var + epsilon) + bias
    return y.astype(x.dtype)


def _batchnorm_training_mode(x, s, bias, mean, var, momentum=0.9,
                             epsilon=1e-5):
    axis = tuple(numpy.delete(numpy.arange(len(x.shape)), 1))
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(x, s, bias, saved_mean, saved_var,
                             epsilon=epsilon)
    return (y.astype(x.dtype), saved_mean.astype(x.dtype),
            saved_var.astype(x.dtype), output_mean.astype(x.dtype),
            output_var.astype(x.dtype))


class BatchNormalization_9(OpRun):

    atts = {'epsilon': 1e-5, 'momentum': 0.9}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=BatchNormalization.atts,
                       **options)

    def _run(self, x, scale, bias, mean, var):  # pylint: disable=W0221
        res = _batchnorm_test_mode(
            x, scale, bias, mean, var, epsilon=self.epsilon)
        return (res, )

    def _infer_shapes(self, x, scale, bias, mean, var):  # pylint: disable=W0221
        return (x, )


class BatchNormalization_14(OpRun):

    atts = {'epsilon': 1e-5, 'momentum': 0.9, 'training_mode': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=BatchNormalization.atts,
                       **options)

    def _run(self, x, scale, bias, mean, var):  # pylint: disable=W0221
        if self.training_mode == 0:
            res = _batchnorm_test_mode(
                x, scale, bias, mean, var, epsilon=self.epsilon)
            return (res, )
        res, saved_mean, saved_var, output_mean, output_var = (
            _batchnorm_training_mode(x, scale, bias, mean, var,
                                     self.momentum, self.epsilon))
        return res, saved_mean, saved_var, output_mean, output_var

    def _infer_shapes(self, x, scale, bias, mean, var):  # pylint: disable=W0221
        if self.training_mode == 0:
            return (x, )
        return (x, scale, bias, mean, var)


if onnx_opset_version() >= 14:
    BatchNormalization = BatchNormalization_14
else:  # pragma: no cover
    BatchNormalization = BatchNormalization_9
