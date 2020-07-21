# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / numpy.sqrt(var + epsilon) + bias


class BatchNormalization(OpRun):

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
