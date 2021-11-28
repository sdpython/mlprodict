"""
@file
@brief Custom operators for gradient numbers.
"""
from skl2onnx.algebra.onnx_operator import OnnxOperator


class OnnxYieldOp_1(OnnxOperator):
    """
    Defines a custom operator for ComplexAbs.
    """

    since_version = 1
    expected_inputs = [('X', 'T')]
    expected_outputs = [('Y', 'T')]
    input_range = [1, 1]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'mlprodict'
    operator_name = 'YieldOp'
    past_version = {}

    def __init__(self, X, non_differentiable_outputs=None,
                 full_shape_outputs=None, op_version=None, **kwargs):
        """
        :param X: array or OnnxOperatorMixin
        :param non_differentiable_outputs: the indices of the module
            outputs that doesn't have a gradient.
        :param full_shape_outputs: the indices of the module outputs
            that must have full shape.
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, X, op_version=op_version, **kwargs)
        self.non_differentiable_outputs = non_differentiable_outputs
        self.full_shape_outputs = full_shape_outputs


OnnxYieldOp = OnnxYieldOp_1
