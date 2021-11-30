"""
@file
@brief Custom operators for complex numbers.
"""
from skl2onnx.algebra.onnx_operator import OnnxOperator


class OnnxComplexAbs_1(OnnxOperator):
    """
    Defines a custom operator for ComplexAbs.
    """

    since_version = 1
    expected_inputs = [('X', 'T'), ]
    expected_outputs = [('Y', 'U')]
    input_range = [1, 1]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'mlprodict'
    operator_name = 'ComplexAbs'
    past_version = {}

    def __init__(self, X, op_version=None, **kwargs):
        """
        :param X: array or OnnxOperatorMixin
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, X, op_version=op_version, **kwargs)


OnnxComplexAbs = OnnxComplexAbs_1
