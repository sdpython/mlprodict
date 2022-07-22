"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import logging
import numpy
from onnx import TensorProto
from onnx.helper import make_attribute
from onnx.numpy_helper import from_array, to_array
from onnx.defs import onnx_opset_version
from skl2onnx.operator_converters.decision_tree import (
    convert_sklearn_decision_tree_regressor,
    convert_sklearn_decision_tree_classifier)
from skl2onnx.operator_converters.gradient_boosting import (
    convert_sklearn_gradient_boosting_regressor,
    convert_sklearn_gradient_boosting_classifier)
from skl2onnx.operator_converters.random_forest import (
    convert_sklearn_random_forest_classifier,
    convert_sklearn_random_forest_regressor_converter)
from skl2onnx.common.data_types import (
    guess_numpy_type, FloatTensorType, DoubleTensorType)


logger = logging.getLogger('mlprodict.onnx_conv')


def _op_type_domain_regressor(dtype, opsetml):
    """
    Defines *op_type* and *op_domain* based on `dtype`.
    """
    if opsetml is None:
        from ... import __max_supported_opsets__
        if onnx_opset_version() >= 16:
            opsetml = min(3, __max_supported_opsets__['ai.onnx.ml'])
        else:
            opsetml = min(1, __max_supported_opsets__['ai.onnx.ml'])
    if opsetml >= 3:
        return 'TreeEnsembleRegressor', 'ai.onnx.ml', 3
    if dtype == numpy.float32:
        return 'TreeEnsembleRegressor', 'ai.onnx.ml', 1
    if dtype == numpy.float64:
        return 'TreeEnsembleRegressorDouble', 'mlprodict', 1
    raise RuntimeError(  # pragma: no cover
        f"Unsupported dtype {dtype}.")


def _op_type_domain_classifier(dtype, opsetml):
    """
    Defines *op_type* and *op_domain* based on `dtype`.
    """
    if opsetml >= 3:
        return 'TreeEnsembleClassifier', 'ai.onnx.ml', 3
    if dtype == numpy.float32:
        return 'TreeEnsembleClassifier', 'ai.onnx.ml', 1
    if dtype == numpy.float64:
        return 'TreeEnsembleClassifierDouble', 'mlprodict', 1
    raise RuntimeError(  # pragma: no cover
        f"Unsupported dtype {dtype}.")


def _fix_tree_ensemble_node(scope, container, opsetml, node, dtype):
    """
    Fixes a node for old versionsof skl2onnx.
    """
    atts = {'base_values': 'base_values_as_tensor',
            'nodes_hitrates': 'nodes_hitrates_as_tensor',
            'nodes_values': 'nodes_values_as_tensor',
            'target_weights': 'target_weights_as_tensor',
            'class_weights': 'class_weights_as_tensor'}
    logger.debug('postprocess %r name=%r opsetml=%r dtype=%r',
                 node.op_type, node.name, opsetml, dtype)
    if dtype == numpy.float64:
        # Inserting a cast operator.
        index = 0 if node.op_type == 'TreeEnsembleRegressor' else 1
        new_name = scope.get_unique_variable_name('tree_ensemble_cast')
        old_name = node.output[index]
        node.output[index] = new_name
        container.add_node(
            'Cast', [new_name], [old_name], to=TensorProto.DOUBLE,  # pylint: disable=E1101
            name=scope.get_unique_operator_name('tree_ensemble_cast'))
    attributes = list(node.attribute)
    del node.attribute[:]
    for att in attributes:
        if att.name in atts:
            logger.debug('+ rewrite att %r into %r', att.name, atts[att.name])
            if att.type == 6:
                value = from_array(
                    numpy.array(att.floats, dtype=dtype), atts[att.name])
            elif att.type == 4:
                value = from_array(
                    numpy.array(att.t.double_data, dtype=dtype), atts[att.name])
            else:
                raise NotImplementedError(
                    "Unable to postprocess attribute name=%r type=%r "
                    "opsetml=%r op_type=%r (value=%r)." % (
                        att.name, att.type, opsetml, node.op_type, att))
            if to_array(value).shape[0] == 0:
                raise RuntimeError(
                    f"Null value from attribute (dtype={dtype!r}): {att!r}.")
            node.attribute.append(make_attribute(atts[att.name], value))
        else:
            node.attribute.append(att)


def _fix_tree_ensemble(scope, container, opsetml, dtype):
    if opsetml is None:
        from ... import __max_supported_opsets__
        if onnx_opset_version() >= 16:
            opsetml = min(3, __max_supported_opsets__['ai.onnx.ml'])
        else:
            opsetml = min(1, __max_supported_opsets__['ai.onnx.ml'])
    if opsetml < 3 or dtype == numpy.float32:
        return False
    for node in container.nodes:
        if node.op_type not in {'TreeEnsembleRegressor', 'TreeEnsembleClassifier'}:
            continue
        _fix_tree_ensemble_node(scope, container, opsetml, node, dtype)
    container.node_domain_version_pair_sets.add(('ai.onnx.ml', opsetml))
    return True


def new_convert_sklearn_decision_tree_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    if opsetml is None:
        opsetml = 3 if container.target_opset >= 16 else 1
    op_type, op_domain, op_version = _op_type_domain_classifier(dtype, opsetml)
    convert_sklearn_decision_tree_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
    _fix_tree_ensemble(scope, container, opsetml, dtype)


def new_convert_sklearn_decision_tree_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    op_type, op_domain, op_version = _op_type_domain_regressor(dtype, opsetml)
    convert_sklearn_decision_tree_regressor(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
    _fix_tree_ensemble(scope, container, opsetml, dtype)


def new_convert_sklearn_gradient_boosting_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    if opsetml is None:
        opsetml = 3 if container.target_opset >= 16 else 1
    op_type, op_domain, op_version = _op_type_domain_classifier(dtype, opsetml)
    convert_sklearn_gradient_boosting_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
    _fix_tree_ensemble(scope, container, opsetml, dtype)


def new_convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    op_type, op_domain, op_version = _op_type_domain_regressor(dtype, opsetml)
    convert_sklearn_gradient_boosting_regressor(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
    _fix_tree_ensemble(scope, container, opsetml, dtype)


def new_convert_sklearn_random_forest_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    if (dtype == numpy.float64 and
            isinstance(operator.outputs[1].type, FloatTensorType)):
        operator.outputs[1].type = DoubleTensorType(
            operator.outputs[1].type.shape)
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    if opsetml is None:
        opsetml = 3 if container.target_opset >= 16 else 1
    op_type, op_domain, op_version = _op_type_domain_classifier(dtype, opsetml)
    convert_sklearn_random_forest_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
    _fix_tree_ensemble(scope, container, opsetml, dtype)


def new_convert_sklearn_random_forest_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    if opsetml is None:
        opsetml = 3 if container.target_opset >= 16 else 1
    op_type, op_domain, op_version = _op_type_domain_regressor(dtype, opsetml)
    convert_sklearn_random_forest_regressor_converter(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)
    _fix_tree_ensemble(scope, container, opsetml, dtype)
