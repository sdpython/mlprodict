"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""

import numpy
from skl2onnx.common._apply_operation import (  # pylint: disable=E0611
    apply_cast, apply_concat, apply_mul,
    apply_reshape, apply_topk
)
from skl2onnx.common.data_types import (
    DoubleTensorType, FloatTensorType
)
from skl2onnx.operator_converters.ada_boost import cum_sum, _apply_gather_elements
from skl2onnx.proto import onnx_proto
from skl2onnx._supported_operators import sklearn_operator_name_map


def _get_estimators_label(scope, operator, container, model):
    """
    This function computes labels for each estimator and returns
    a tensor produced by concatenating the labels.
    """
    var_type = (FloatTensorType if container.proto_dtype == numpy.float32
                else DoubleTensorType)
    concatenated_labels_name = scope.get_unique_variable_name(
        'concatenated_labels')

    input_name = operator.inputs
    estimators_results_list = []
    for i, estimator in enumerate(model.estimators_):
        estimator_label_name = scope.declare_local_variable(
            'est_label_%d' % i, var_type([None, 1]))

        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type)
        this_operator.raw_operator = estimator
        this_operator.inputs = input_name
        this_operator.outputs.append(estimator_label_name)

        estimators_results_list.append(estimator_label_name.onnx_name)

    apply_concat(scope, estimators_results_list, concatenated_labels_name,
                 container, axis=1)
    return concatenated_labels_name


def convert_sklearn_ada_boost_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supported
    doubles.
    """
    op = operator.raw_operator

    negate_name = scope.get_unique_variable_name('negate')
    estimators_weights_name = scope.get_unique_variable_name(
        'estimators_weights')
    half_scalar_name = scope.get_unique_variable_name('half_scalar')
    last_index_name = scope.get_unique_variable_name('last_index')
    negated_labels_name = scope.get_unique_variable_name('negated_labels')
    sorted_values_name = scope.get_unique_variable_name('sorted_values')
    sorted_indices_name = scope.get_unique_variable_name('sorted_indices')
    array_feat_extractor_output_name = scope.get_unique_variable_name(
        'array_feat_extractor_output')
    median_value_name = scope.get_unique_variable_name('median_value')
    comp_value_name = scope.get_unique_variable_name('comp_value')
    median_or_above_name = scope.get_unique_variable_name('median_or_above')
    median_idx_name = scope.get_unique_variable_name('median_idx')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    reshaped_weights_name = scope.get_unique_variable_name('reshaped_weights')
    median_estimators_name = scope.get_unique_variable_name(
        'median_estimators')

    container.add_initializer(negate_name, container.proto_dtype,
                              [], [-1])
    container.add_initializer(estimators_weights_name,
                              container.proto_dtype,
                              [len(op.estimator_weights_)],
                              op.estimator_weights_)
    container.add_initializer(half_scalar_name, container.proto_dtype,
                              [], [0.5])
    container.add_initializer(last_index_name, onnx_proto.TensorProto.INT64,  # pylint: disable=E1101
                              [], [len(op.estimators_) - 1])

    concatenated_labels = _get_estimators_label(scope, operator,
                                                container, op)
    apply_mul(scope, [concatenated_labels, negate_name],
              negated_labels_name, container, broadcast=1)
    try:
        apply_topk(scope, negated_labels_name,
                   [sorted_values_name, sorted_indices_name],
                   container, k=len(op.estimators_))
    except TypeError:
        # issue with onnxconverter-common
        apply_topk(scope, [negated_labels_name],
                   [sorted_values_name, sorted_indices_name],
                   container, k=len(op.estimators_))
    container.add_node(
        'ArrayFeatureExtractor',
        [estimators_weights_name, sorted_indices_name],
        array_feat_extractor_output_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_reshape(
        scope, array_feat_extractor_output_name, reshaped_weights_name,
        container, desired_shape=(-1, len(op.estimators_)))
    weights_cdf_name = cum_sum(
        scope, container, reshaped_weights_name,
        len(op.estimators_))
    container.add_node(
        'ArrayFeatureExtractor', [weights_cdf_name, last_index_name],
        median_value_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_mul(scope, [median_value_name, half_scalar_name],
              comp_value_name, container, broadcast=1)
    container.add_node(
        'Less', [weights_cdf_name, comp_value_name],
        median_or_above_name,
        name=scope.get_unique_operator_name('Less'))
    apply_cast(scope, median_or_above_name, cast_result_name,
               container, to=container.proto_dtype)
    container.add_node('ArgMin', cast_result_name,
                       median_idx_name,
                       name=scope.get_unique_operator_name('ArgMin'), axis=1)
    _apply_gather_elements(
        scope, container, [sorted_indices_name, median_idx_name],
        median_estimators_name, axis=1, dim=len(op.estimators_),
        zero_type=onnx_proto.TensorProto.INT64, suffix="A")  # pylint: disable=E1101
    output_name = operator.output_full_names[0]
    _apply_gather_elements(
        scope, container, [concatenated_labels, median_estimators_name],
        output_name, axis=1, dim=len(op.estimators_),
        zero_type=container.proto_dtype, suffix="B")
