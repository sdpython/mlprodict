# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
    # Inspired from: https://github.com/onnx/onnx/blob/main/onnx/backend/
    # test/case/node/layernormalization.py#L12
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = numpy.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = numpy.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = numpy.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = numpy.sqrt(variance_eps)
    inv_std_dev = numpy.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = numpy.reshape(y_mat, X_shape) * W
    if B is not None:
        Y = Y + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = numpy.reshape(x_mean, reduction_shape)
    X_inv_std_dev = numpy.reshape(inv_std_dev, reduction_shape)

    return (Y.astype(X.dtype),
            X_mean.astype(X.dtype),
            X_inv_std_dev.astype(X.dtype))


class LayerNormalization(OpRun):

    atts = {'axis': -1,
            'epsilon': 9.999999747378752e-06,
            'stash_type': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=LayerNormalization.atts,
                       **options)

    def _run(self, X, Scale, B=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        res = _layer_normalization(
            X, Scale, B, axis=self.axis, epsilon=self.epsilon)
        return res
