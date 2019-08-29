"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import numpy
from skl2onnx.algebra.complex_functions import onnx_cdist
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxTopK, OnnxMul, OnnxArrayFeatureExtractor, OnnxReduceMean,  # pylint: disable=E0611
    OnnxFlatten, OnnxShape, OnnxReshape  # pylint: disable=E0611
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
    if not single_reg:
        raise NotImplementedError("Only single regression is implemented.")

    # ndim = 1 if single_reg else op._y.shape[1]

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
    extracted = OnnxArrayFeatureExtractor(
        training_labels.ravel(), flattened, op_version=opv)
    reshaped = OnnxReshape(extracted, shape, op_version=opv)
    res = OnnxReduceMean(reshaped, axes=[1], op_version=opv,
                         output_names=out)
    res.add_to(scope, container)
