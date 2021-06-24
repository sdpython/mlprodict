"""
@file
@brief Implementation of a dummy score using
:epkg:`cdist`.
"""
import numpy
from onnx import onnx_pb as onnx_proto
from scipy.spatial.distance import cdist


def score_cdist_sum(expected_values, predictions,
                    metric='sqeuclidean', p=None):
    """
    Computes the sum of pairwise distances between
    *expected_values* and *predictions*. It has no
    particular purpose except the one of converting
    a scorer into ONNX.

    @param      expected_values     expected_values
    @param      predictions         predictions
    @param      metric              see function :epkg:`cdist`
    @param      p                   see function :epkg:`cdist`
    @return                         some of the pairwise distances
    """
    if p is None:
        dist = cdist(expected_values, predictions, metric=metric)
    else:
        dist = cdist(expected_values, predictions, metric=metric, p=p)
    return numpy.sum(dist, axis=1)


def convert_score_cdist_sum(scope, operator, container):
    """
    Converts function @see fn score_cdist_sum into :epkg:`ONNX`.
    """
    op = operator.raw_operator
    if op._fct != score_cdist_sum:  # pylint: disable=W0143
        raise RuntimeError(  # pragma: no cover
            "The wrong converter was called {} != {}.".format(
                op._fct, score_cdist_sum))

    from skl2onnx.algebra.complex_functions import onnx_cdist
    from skl2onnx.algebra.onnx_ops import OnnxReduceSumApi11  # pylint: disable=E0611
    from skl2onnx.common.data_types import guess_numpy_type

    X = operator.inputs[0]
    Y = operator.inputs[1]
    out = operator.outputs
    opv = container.target_opset
    dtype = guess_numpy_type(operator.inputs[0].type)
    out = operator.outputs

    options = container.get_options(score_cdist_sum, dict(cdist=None))

    kwargs = op.kwargs

    if options.get('cdist', None) == 'single-node':
        attrs = kwargs
        cdist_name = scope.get_unique_variable_name('cdist')
        container.add_node('CDist', [X.full_name, Y.full_name], cdist_name,
                           op_domain='mlprodict', name=scope.get_unique_operator_name('CDist'),
                           **attrs)
        if container.target_opset < 13:
            container.add_node('ReduceSum', [cdist_name], out[0].full_name,
                               axes=[1], keepdims=0,
                               name=scope.get_unique_operator_name('ReduceSum'))
        else:
            axis_name = scope.get_unique_variable_name('axis')
            container.add_initializer(
                axis_name, onnx_proto.TensorProto.INT64, [1], [1])  # pylint: disable=E1101
            container.add_node(
                'ReduceSum', [cdist_name, axis_name],
                out[0].full_name, keepdims=0,
                name=scope.get_unique_operator_name('ReduceSum'))
    else:
        metric = kwargs['metric']
        if metric == 'minkowski':
            dists = onnx_cdist(X, Y, dtype=dtype, op_version=opv,
                               metric=metric, p=kwargs.get('p', 2))
        else:
            dists = onnx_cdist(X, Y, dtype=dtype, op_version=opv,
                               metric=kwargs['metric'])

        res = OnnxReduceSumApi11(dists, axes=[1], keepdims=0,
                                 output_names=[out[0].full_name],
                                 op_version=opv)
        res.add_to(scope, container)
