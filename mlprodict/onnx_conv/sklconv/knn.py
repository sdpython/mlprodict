"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import numpy
from skl2onnx.algebra.complex_functions import onnx_cdist
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxTopK, OnnxMul, OnnxArrayFeatureExtractor, OnnxReduceMean,
    OnnxFlatten, OnnxShape, OnnxReshape,
    OnnxConcat, OnnxTranspose
)


def onnx_nearest_neighbors_indices(X, Y, k, metric='euclidean', dtype=None, **kwargs):
    """
    Retrieves the nearest neigbours :epkg:`ONNX`.
    :param X: features
    :param Y: neighbours
    :param k: number of neighbours to retrieve
    :param metric: requires metric
    :param dtype: numerical type
    :param kwargs: additional parameters such as *op_version*
    :return: top indices
    """
    dist = onnx_cdist(X, Y, metric=metric, dtype=dtype, **kwargs)
    neg_dist = OnnxMul(dist, numpy.array([-1], dtype=dtype))
    topk = OnnxTopK(neg_dist, numpy.array([k], dtype=numpy.int64),
                    **kwargs)[1]
    return topk


def convert_nearest_neighbors_regressor(scope, operator, container):
    """
    Converts :epkg:`sklearn:neighbors:KNeighborsRegressor` into
    :epkg:`ONNX`.
    """
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    dtype = container.dtype
    out = operator.outputs

    single_reg = (len(op._y.shape) == 1 or len(
        op._y.shape) == 2 and op._y.shape[1] == 1)
    ndim = 1 if single_reg else op._y.shape[1]

    metric = op.effective_metric_
    neighb = op._fit_X.astype(container.dtype)
    k = op.n_neighbors
    training_labels = op._y
    # distance_power = (
    #     op.p if op.metric == 'minkowski'
    #     else (2 if op.metric in ('euclidean', 'l2') else 1))

    top_indices = onnx_nearest_neighbors_indices(
        X, neighb, k, metric=metric, dtype=dtype, op_version=opv)
    shape = OnnxShape(top_indices, op_version=opv)
    flattened = OnnxFlatten(top_indices, op_version=opv)
    if ndim > 1:
        # shape = (ntargets, ) + shape
        training_labels = training_labels.T
        shape = OnnxConcat(numpy.array([ndim], dtype=numpy.int64),
                           shape, op_version=opv)
        axis = 2
    else:
        training_labels = training_labels.ravel()
        axis = 1
    extracted = OnnxArrayFeatureExtractor(
        training_labels, flattened, op_version=opv)
    reshaped = OnnxReshape(extracted, shape, op_version=opv)
    if ndim > 1:
        reshaped = OnnxTranspose(reshaped, op_version=opv, perm=[1, 0, 2])

    res = OnnxReduceMean(reshaped, axes=[axis], op_version=opv,
                         keepdims=0, output_names=out)
    res.add_to(scope, container)
