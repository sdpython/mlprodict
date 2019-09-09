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
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_regressor_attribute_pairs
)
from skl2onnx.operator_converters.ada_boost import cum_sum
from skl2onnx.proto import onnx_proto


def _get_estimators_label(scope, operator, container, model):
    """
    This function computes labels for each estimator and returns
    a tensor produced by concatenating the labels.
    """
    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleRegressor'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleRegressorDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    concatenated_labels_name = scope.get_unique_variable_name(
        'concatenated_labels')

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')
        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=container.proto_dtype)
        input_name = cast_input_name

    estimators_results_list = []
    for tree_id in range(len(model.estimators_)):
        estimator_label_name = scope.get_unique_variable_name(
            'estimator_label')
        attrs = get_default_tree_regressor_attribute_pairs()
        attrs['name'] = scope.get_unique_operator_name(op_type)
        attrs['n_targets'] = int(model.estimators_[tree_id].n_outputs_)
        add_tree_to_attribute_pairs(
            attrs, False, model.estimators_[tree_id].tree_, 0, 1, 0, False,
            True, dtype=container.dtype)
        container.add_node(op_type, input_name, estimator_label_name,
                           op_domain=op_domain, **attrs)
        estimators_results_list.append(estimator_label_name)
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

    container.add_initializer(negate_name, container.proto_dtype, [], [-1])
    container.add_initializer(estimators_weights_name, container.proto_dtype,
                              [len(op.estimator_weights_)], op.estimator_weights_)
    container.add_initializer(
        half_scalar_name, container.proto_dtype, [], [0.5])
    container.add_initializer(last_index_name, onnx_proto.TensorProto.INT64,  # pylint: disable=E1101
                              [], [len(op.estimators_) - 1])

    concatenated_labels = _get_estimators_label(scope, operator, container, op)
    apply_mul(scope, [concatenated_labels, negate_name],
              negated_labels_name, container, broadcast=1)
    apply_topk(scope, negated_labels_name,
               [sorted_values_name, sorted_indices_name],
               container, k=len(op.estimators_))
    container.add_node('ArrayFeatureExtractor', [estimators_weights_name, sorted_indices_name],
                       array_feat_extractor_output_name, op_domain='ai.onnx.ml',
                       name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_reshape(scope, array_feat_extractor_output_name, reshaped_weights_name,
                  container, desired_shape=(-1, len(op.estimators_)))
    weights_cdf_name = cum_sum(
        scope, container, reshaped_weights_name, len(op.estimators_))
    container.add_node('ArrayFeatureExtractor', [weights_cdf_name, last_index_name],
                       median_value_name, op_domain='ai.onnx.ml',
                       name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_mul(scope, [median_value_name, half_scalar_name],
              comp_value_name, container, broadcast=1)
    container.add_node('Less', [weights_cdf_name, comp_value_name], median_or_above_name,
                       name=scope.get_unique_operator_name('Less'))
    apply_cast(scope, median_or_above_name, cast_result_name,
               container, to=container.proto_dtype)
    container.add_node('ArgMin', cast_result_name, median_idx_name,
                       name=scope.get_unique_operator_name('ArgMin'), axis=1)
    container.add_node(
        'ArrayFeatureExtractor', [sorted_indices_name, median_idx_name],
        median_estimators_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    container.add_node(
        'ArrayFeatureExtractor', [concatenated_labels, median_estimators_name],
        operator.output_full_names, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
