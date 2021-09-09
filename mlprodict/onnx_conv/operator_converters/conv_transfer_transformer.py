"""
@file
@brief Converters for models from :epkg:`mlinsights`.
"""
from sklearn.base import ClassifierMixin
from skl2onnx import get_model_alias
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common._registration import (
    get_shape_calculator, _converter_pool, _shape_calculator_pool)
from skl2onnx._parse import parse_sklearn
from skl2onnx.common._apply_operation import apply_identity
from skl2onnx.common._topology import Scope, Variable
from skl2onnx._supported_operators import sklearn_operator_name_map


def _model_outputs(scope, model, inputs, custom_parsers=None):
    """
    Retrieves the outputs of one particular models.
    """
    if custom_parsers is not None and model in custom_parsers:
        return custom_parsers[model](
            scope, model, inputs, custom_parsers=custom_parsers)
    return parse_sklearn(scope, model, inputs, custom_parsers=custom_parsers)


def parser_transfer_transformer(scope, model, inputs, custom_parsers=None):
    """
    Parser for :epkg:`TransferTransformer`.
    """
    if custom_parsers is not None and model in custom_parsers:
        return custom_parsers[model](
            scope, model, inputs, custom_parsers=custom_parsers)

    if model.method == 'predict_proba':
        name = 'probabilities'
    elif model.method == 'transform':
        name = 'variable'
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unable to defined the output for method='{}' and model='{}'.".format(
                model.method, model.__class__.__name__))

    prob = scope.declare_local_variable(name, FloatTensorType())
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs
    this_operator.outputs.append(prob)
    return this_operator.outputs


def shape_calculator_transfer_transformer(operator):
    """
    Shape calculator :epkg:`TransferTransformer`.
    """
    op = operator.raw_operator
    alias = get_model_alias(type(op.estimator_))
    calc = get_shape_calculator(alias)

    options = (None if not hasattr(operator.scope, 'options')
               else operator.scope.options)
    registered_models = dict(
            conv=_converter_pool, shape=_shape_calculator_pool,
            aliases=sklearn_operator_name_map)
    scope = Scope('temp', options=options,
                  registered_models=registered_models)
    this_operator = scope.declare_local_operator(alias)
    this_operator.raw_operator = op.estimator_
    this_operator.inputs = [
        Variable(v.onnx_name, v.onnx_name, type=v.type, scope=scope)
        for v in operator.inputs]
    res = _model_outputs(scope, op.estimator_, this_operator.inputs)
    this_operator.outputs.extend([
        scope.declare_local_variable(
            "%sTTS" % r.onnx_name, r.type) for r in res])
    this_operator.outputs = res
    calc(this_operator)

    if op.method == 'predict_proba':
        operator.outputs[0].type = this_operator.outputs[1].type
    elif op.method == 'transform':
        operator.outputs[0].type = this_operator.outputs[0].type
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unable to defined the output for method='{}' and model='{}'.".format(
                op.method, op.__class__.__name__))


def convert_transfer_transformer(scope, operator, container):
    """
    Converters for :epkg:`TransferTransformer`.
    """
    op = operator.raw_operator
    op_type = get_model_alias(type(op.estimator_))

    this_operator = scope.declare_local_operator(op_type)
    this_operator.raw_operator = op.estimator_
    this_operator.inputs = operator.inputs

    if isinstance(op.estimator_, ClassifierMixin):
        container.add_options(id(op.estimator_), {'zipmap': False})

    res = _model_outputs(scope, op.estimator_, operator.inputs)
    this_operator.outputs.extend([
        scope.declare_local_variable(
            "%sTTC" % r.onnx_name, r.type) for r in res])

    if op.method == 'predict_proba':
        index = 1
    elif op.method == 'transform':
        index = 0
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unable to defined the output for method='{}' and model='{}'.".format(
                op.method, op.__class__.__name__))

    apply_identity(scope, this_operator.outputs[index].onnx_name,
                   operator.outputs[0].full_name, container,
                   operator_name=scope.get_unique_operator_name("IdentityTT"))
