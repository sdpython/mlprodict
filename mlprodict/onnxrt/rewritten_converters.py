"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import numpy
from skl2onnx.common._apply_operation import apply_cast
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.common._registration import _converter_pool
from skl2onnx.common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_regressor_attribute_pairs
)


def convert_sklearn_decision_tree_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supported
    doubles.
    """
    op = operator.raw_operator
    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleRegressor'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleRegressorDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = int(op.n_outputs_)
    add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=container.proto_type)
        input_name = cast_input_name

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain=op_domain,
                       **attrs)


def convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supported
    doubles.
    """
    op = operator.raw_operator

    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleRegressor'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleRegressorDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1

    if op.init == 'zero':
        cst = numpy.zeros(op.loss_.K)
    elif op.init is None:
        # constant_ was introduced in scikit-learn 0.21.
        if hasattr(op.init_, 'constant_'):
            cst = [float(x) for x in op.init_.constant_]
        elif op.loss == 'ls':
            cst = [op.init_.mean]
        else:
            cst = [op.init_.quantile]
    else:
        raise NotImplementedError(
            'Setting init to an estimator is not supported, you may raise an '
            'issue at https://github.com/onnx/sklearn-onnx/issues.')

    attrs['base_values'] = [float(x) for x in cst]

    tree_weight = op.learning_rate
    n_est = (op.n_estimators_ if hasattr(op, 'n_estimators_') else
             op.n_estimators)
    for i in range(n_est):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight,
                                    0, False)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=container.proto_type)
        input_name = cast_input_name

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain=op_domain,
                       **attrs)


_overwritten_operators = {
    'SklearnDecisionTreeRegressor': convert_sklearn_decision_tree_regressor,
    'SklearnGradientBoostingRegressor': convert_sklearn_gradient_boosting_regressor,
}


def register_rewritten_operators(new_values=None):
    """
    Registers modified operators and returns the old values.

    @param      new_values      operators to rewrite or None
                                to rewrite default ones
    @return                      old values
    """
    if new_values is None:
        for rew in _overwritten_operators:
            if rew not in _converter_pool:
                raise KeyError(
                    "skl2onnx was not imported and '{}' was not registered.".format(rew))
        old_values = {k: _converter_pool[k] for k in _overwritten_operators}
        _converter_pool.update(_overwritten_operators)
        return old_values
    else:
        for rew in new_values:
            if rew not in _converter_pool:
                raise KeyError(
                    "skl2onnx was not imported and '{}' was not registered.".format(rew))
        old_values = {k: _converter_pool[k] for k in new_values}
        _converter_pool.update(new_values)
        return old_values
