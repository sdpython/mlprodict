"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
from collections import OrderedDict
import numpy
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxTopK, OnnxMul, OnnxArrayFeatureExtractor, OnnxReduceMean,
    OnnxFlatten, OnnxShape, OnnxReshape,
    OnnxConcat, OnnxTranspose, OnnxSub,
    OnnxIdentity, OnnxReduceSumSquare,
    OnnxScan, OnnxSqrt, OnnxReciprocal,
    OnnxPow, OnnxReduceSum, OnnxAbs,
    OnnxMax, OnnxDiv
)


def onnx_cdist(X, Y, metric='sqeuclidean', dtype=None, op_version=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, Y, metric=metric)``.

    :param X: :epkg:`numpy:ndarray` or :epkg:`OnnxOperatorMixin`
    :param Y: :epkg:`numpy:ndarray` or :epkg:`OnnxOperatorMixin`
    :param metric: distance type
    :param dtype: *numpy.float32* or *numpy.float64*
    :param op_version: opset version
    :param kwargs: addition parameter
    :return: :epkg:`OnnxOperatorMixin`
    """
    if metric == 'sqeuclidean':
        return _onnx_cdist_sqeuclidean(
            X, Y, dtype=dtype, op_version=op_version, **kwargs)
    elif metric == 'euclidean':
        res = _onnx_cdist_sqeuclidean(X, Y, dtype=dtype, op_version=op_version)
        return OnnxSqrt(res, op_version=op_version, **kwargs)
    elif metric == 'minkowski':
        p = kwargs.pop('p')
        res = _onnx_cdist_minkowski(
            X, Y, dtype=dtype, op_version=op_version, p=p)
        return OnnxPow(res, numpy.array([1. / p], dtype=dtype),
                       op_version=op_version, **kwargs)
    elif metric == 'manhattan':
        return _onnx_cdist_manhattan(
            X, Y, dtype=dtype, op_version=op_version, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def _onnx_cdist_sqeuclidean(X, Y, dtype=None, op_version=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, metric='sqeuclidean')``.
    """
    diff = OnnxSub('next_in', 'next', output_names=[
                   'diff'], op_version=op_version)
    id_next = OnnxIdentity('next_in', output_names=[
                           'next_out'], op_version=op_version)
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[
                               1], keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=['scan_out'], op_version=op_version)
    tensor_type = FloatTensorType if dtype == numpy.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('cdistsqe')
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', tensor_type()),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        dtype=dtype, target_opset=op_version)

    node = OnnxScan(X, Y, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph, op_version=op_version)
    return OnnxTranspose(node[1], perm=[1, 0], op_version=op_version,
                         **kwargs)


def _onnx_cdist_minkowski(X, Y, dtype=None, op_version=None, p=2, **kwargs):
    """
    Returns the ONNX graph which computes the :epkg:`Minkowski distance`
    or ``minkowski(X, Y, p)``.
    """
    diff = OnnxSub('next_in', 'next', output_names=[
                   'diff'], op_version=op_version)
    id_next = OnnxIdentity('next_in', output_names=[
                           'next_out'], op_version=op_version)
    diff_pow = OnnxPow(OnnxAbs(diff, op_version=op_version),
                       numpy.array([p], dtype=dtype), op_version=op_version)
    norm = OnnxReduceSum(diff_pow, axes=[1], output_names=[
                         'norm'], keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=['scan_out'], op_version=op_version)
    tensor_type = FloatTensorType if dtype == numpy.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('cdistmink')
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', tensor_type()),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        dtype=dtype, target_opset=op_version)

    node = OnnxScan(X, Y, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph, op_version=op_version)
    return OnnxTranspose(node[1], perm=[1, 0], op_version=op_version,
                         **kwargs)


def _onnx_cdist_manhattan(X, Y, dtype=None, op_version=None, **kwargs):
    """
    Returns the ONNX graph which computes the :epkg:`Minkowski distance`
    or ``minkowski(X, Y, p)``.
    """
    diff = OnnxSub('next_in', 'next', output_names=[
                   'diff'], op_version=op_version)
    id_next = OnnxIdentity('next_in', output_names=[
                           'next_out'], op_version=op_version)
    diff_pow = OnnxAbs(diff, op_version=op_version)
    norm = OnnxReduceSum(diff_pow, axes=[1], output_names=[
                         'norm'], keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=['scan_out'], op_version=op_version)
    tensor_type = FloatTensorType if dtype == numpy.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('cdistmink')
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', tensor_type()),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        dtype=dtype, target_opset=op_version)

    node = OnnxScan(X, Y, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph, op_version=op_version)
    return OnnxTranspose(node[1], perm=[1, 0], op_version=op_version,
                         **kwargs)


def onnx_nearest_neighbors_indices(X, Y, k, metric='euclidean', dtype=None,
                                   op_version=None, keep_distances=False,
                                   optim=None, **kwargs):
    """
    Retrieves the nearest neigbours :epkg:`ONNX`.
    :param X: features or :epkg:`OnnxOperatorMixin`
    :param Y: neighbours or :epkg:`OnnxOperatorMixin`
    :param k: number of neighbours to retrieve
    :param metric: requires metric
    :param dtype: numerical type
    :param op_version: opset version
    :param keep_distance: returns the distances as well (second position)
    :param optim: implements specific optimisations,
        ``'cdist'`` replaces *Scan* operator by operator *CDist*
    :param kwargs: additional parameters for function @see fn onnx_cdist
    :return: top indices
    """
    if optim == 'cdist':
        from skl2onnx.algebra.custom_ops import OnnxCDist
        dist = OnnxCDist(X, Y, metric=metric, op_version=op_version,
                         **kwargs)
    elif optim is None:
        dist = onnx_cdist(X, Y, metric=metric, dtype=dtype,
                          op_version=op_version, **kwargs)
    else:
        raise ValueError("Unknown optimisation '{}'.".format(optim))
    neg_dist = OnnxMul(dist, numpy.array(
        [-1], dtype=dtype), op_version=op_version)
    node = OnnxTopK(neg_dist, numpy.array([k], dtype=numpy.int64),
                    op_version=op_version, **kwargs)
    if keep_distances:
        return (node[1], OnnxMul(node[0], numpy.array(
                    [-1], dtype=dtype), op_version=op_version))
    else:
        return node[1]


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

    options = container.get_options(op, dict(optim=None))

    single_reg = (len(op._y.shape) == 1 or len(
        op._y.shape) == 2 and op._y.shape[1] == 1)
    ndim = 1 if single_reg else op._y.shape[1]

    metric = op.effective_metric_
    neighb = op._fit_X.astype(container.dtype)
    k = op.n_neighbors
    training_labels = op._y
    distance_kwargs = {}
    if metric == 'minkowski':
        if op.p != 2:
            distance_kwargs['p'] = op.p
        else:
            metric = "euclidean"

    if op.weights == 'uniform':
        top_indices = onnx_nearest_neighbors_indices(
            X, neighb, k, metric=metric, dtype=dtype,
            op_version=opv, optim=options.get('optim', None),
            **distance_kwargs)
        top_distances = None
    elif op.weights == 'distance':
        top_indices, top_distances = onnx_nearest_neighbors_indices(
            X, neighb, k, metric=metric, dtype=dtype,
            op_version=opv, keep_distances=True,
            optim=options.get('optim', None),
            **distance_kwargs)
    else:
        raise RuntimeError(
            "Unable to convert KNeighborsRegressor when weights is callable.")

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

    if top_distances is not None:
        modified = OnnxMax(top_distances, numpy.array([1e-6], dtype=dtype),
                           op_version=opv)
        wei = OnnxReciprocal(modified, op_version=opv)
        norm = OnnxReduceSum(wei, op_version=opv, axes=[1], keepdims=0)
        weighted = OnnxMul(reshaped, wei, op_version=opv)
        res = OnnxReduceSum(weighted, axes=[axis], op_version=opv,
                            keepdims=0)
        res = OnnxDiv(res, norm, op_version=opv, output_names=out)
    else:
        res = OnnxReduceMean(reshaped, axes=[axis], op_version=opv,
                             keepdims=0, output_names=out)
    res.add_to(scope, container)
