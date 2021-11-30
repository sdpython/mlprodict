"""
@file
@brief Custom operators for gradient numbers.
"""
from skl2onnx.algebra.onnx_operator import OnnxOperator


class OnnxYieldOp_1(OnnxOperator):
    """
    Defines a custom operator for YieldOp.
    """

    since_version = 1
    expected_inputs = [('X', 'T')]
    expected_outputs = [('Y', 'T')]
    input_range = [1, 1]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'com.microsoft'
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


class OnnxBroadcastGradientArgs_1(OnnxOperator):
    """
    Defines a custom operator for BroadcastGradientArgs.
    Returns the reduction axes for computing gradients of s0 op s1 with
    broadcast. The ouput axes are deterministic from last to first.
    Output is an empty vector when no reduction is necessary for the
    corresponding input.
    """

    since_version = 1
    expected_inputs = [('a_shape', 'T'), ('b_shape', 'T')]
    expected_outputs = [('a_axes', 'T'), ('b_axes', 'T')]
    input_range = [2, 2]
    output_range = [2, 2]
    is_deprecated = False
    domain = 'com.microsoft'
    operator_name = 'BroadcastGradientArgs'
    past_version = {}

    def __init__(self, a_shape, b_shape, op_version=None, **kwargs):
        """
        :param a_shape: The 1st input shape as Tensor.
        :param b_shape: The 2nds input shape as Tensor.
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, a_shape, b_shape, op_version=op_version, **kwargs)


OnnxBroadcastGradientArgs = OnnxBroadcastGradientArgs_1


class OnnxFusedMatMul_1(OnnxOperator):
    """
    MatMul and Gemm without a C.
    """

    since_version = 1
    expected_inputs = [('X', 'T'), ('X', 'T')]
    expected_outputs = [('Z', 'T')]
    input_range = [2, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'com.microsoft'
    operator_name = 'FusedMatMul'
    past_version = {}

    def __init__(self, X, Y, transA=0, transB=0,
                 op_version=None, **kwargs):
        """
        :param X: first matrix
        :param Y: second matrix
        :param transA: transpose first matrix
        :param transB: transpose second matrix
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, X, Y, transA=transA, transB=transB,
            op_version=op_version, **kwargs)


OnnxFusedMatMul = OnnxFusedMatMul_1


class OnnxSoftmaxGrad_13(OnnxOperator):
    """
    Gradient of Softmax.
    SoftmaxGrad computes :math:`Y * ( dY - ReduceSum(Y * dY))`.
    ONNX does not have a dot product,
    which can be simulated as a pointwise-multiplication ("Mul"),
    followed by a "ReduceSum". Unfortunately, the treatment of "axis"
    is different in "SoftmaxGrad" and "ReduceSum".
    If axis=k for SoftmaxGrad, we need to specify [k, ..., n-1] as the axes of
    reduction for "ReduceSum", after accounting for negative-axis specification.
    An alternative solution would be to Flatten inputs to 2D and then reshape
    output back to original shape. Hopefully, many of these ops can be optimized
    away in the common-case of statically-known shapes.
    """

    since_version = 1
    expected_inputs = [('grad', 'T'), ('prob', 'T')]
    expected_outputs = [('Y', 'T')]
    input_range = [2, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'com.microsoft'
    operator_name = 'SoftmaxGrad_13'
    past_version = {}

    def __init__(self, grad, prob, op_version=None, **kwargs):
        """
        :param grad: gradient
        :param prob: probablities
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, grad, prob, op_version=op_version, **kwargs)


OnnxSoftmaxGrad = OnnxSoftmaxGrad_13
