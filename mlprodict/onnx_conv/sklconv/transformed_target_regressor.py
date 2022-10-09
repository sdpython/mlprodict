"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
from sklearn.preprocessing import FunctionTransformer
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator


def transformer_target_regressor_shape_calculator(operator):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support custom functions
    implemented with :ref:`l-numpy-onnxpy`.
    """
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    # same output shape as input
    output_type = input_type([None, None])
    operator.outputs[0].type = output_type


def transformer_target_regressor_converter(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support custom functions
    implemented with :ref:`l-numpy-onnxpy`.
    """
    op = operator.raw_operator
    opv = container.target_opset
    X = operator.inputs[0]

    Y = OnnxSubEstimator(op.regressor_, X, op_version=opv)
    cpy = FunctionTransformer(op.transformer_.inverse_func)
    Z = OnnxSubEstimator(cpy, Y, output_names=operator.outputs)
    Z.add_to(scope, container)
