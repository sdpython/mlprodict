"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import numpy
from skl2onnx.operator_converters.decision_tree import (
    convert_sklearn_decision_tree_regressor,
    convert_sklearn_decision_tree_classifier)
from skl2onnx.operator_converters.gradient_boosting import (
    convert_sklearn_gradient_boosting_regressor,
    convert_sklearn_gradient_boosting_classifier)
from skl2onnx.operator_converters.random_forest import (
    convert_sklearn_random_forest_classifier,
    convert_sklearn_random_forest_regressor_converter)


def _op_type_domain_regressor(container):
    """
    Defines *op_type* and *op_domain* based on
    `container.dtype`.
    """
    if container.dtype == numpy.float32:
        return 'TreeEnsembleRegressor', 'ai.onnx.ml', 1
    if container.dtype == numpy.float64:
        return 'TreeEnsembleRegressorDouble', 'mlprodict', 1
    raise RuntimeError("Unsupported dtype {}.".format(container.dtype))


def _op_type_domain_classifier(container):
    """
    Defines *op_type* and *op_domain* based on
    `container.dtype`.
    """
    if container.dtype == numpy.float32:
        return 'TreeEnsembleClassifier', 'ai.onnx.ml', 1
    if container.dtype == numpy.float64:
        return 'TreeEnsembleClassifierDouble', 'mlprodict', 1
    raise RuntimeError("Unsupported dtype {}.".format(container.dtype))


def new_convert_sklearn_decision_tree_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op_type, op_domain, op_version = _op_type_domain_classifier(container)
    convert_sklearn_decision_tree_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def new_convert_sklearn_decision_tree_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op_type, op_domain, op_version = _op_type_domain_regressor(container)
    convert_sklearn_decision_tree_regressor(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def new_convert_sklearn_gradient_boosting_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op_type, op_domain, op_version = _op_type_domain_classifier(container)
    convert_sklearn_gradient_boosting_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def new_convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op_type, op_domain, op_version = _op_type_domain_regressor(container)
    convert_sklearn_gradient_boosting_regressor(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def new_convert_sklearn_random_forest_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op_type, op_domain, op_version = _op_type_domain_classifier(container)
    convert_sklearn_random_forest_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def new_convert_sklearn_random_forest_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op_type, op_domain, op_version = _op_type_domain_regressor(container)
    convert_sklearn_random_forest_regressor_converter(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
