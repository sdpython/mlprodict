"""
@file
@brief Computes shape inference for element wise operators with one input.
"""
import numpy
from .shape_excs import ShapeInferenceException
from .shape_result import OnnxKind


def _element_unary(known_shapes, node, dtype=None, one_input=True):
    """
    Infers shape for an element wise operator.
    The function returns but updates *known_shapes*.

    :param known_shapes: known shapes
    :param node: Onnx node
    :param dtype: None to keep the same type as input,
        not None to change it
    :param one_input: check there is only one input
    :return: updated or not
    """
    if one_input and len(node.input) != 1:
        raise ShapeInferenceException(  # pragma: no cover
            f"Node {node.name!r} must have one input not {len(node.input)}.")
    x = known_shapes[node.input[0]]
    if x.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            f"Result {x!r} must be a tensor.")
    if dtype is None:
        return known_shapes.update(node.output[0], x.copy())
    cp = x.copy()
    cp.dtype = dtype
    return known_shapes.update(node.output[0], cp)


def shape_abs(known_shapes, node):
    "Infers shape for operator Abs."
    return _element_unary(known_shapes, node)


def shape_acos(known_shapes, node):
    "Infers shape for operator Acos."
    return _element_unary(known_shapes, node)


def shape_acosh(known_shapes, node):
    "Infers shape for operator Acosh."
    return _element_unary(known_shapes, node)


def shape_asin(known_shapes, node):
    "Infers shape for operator Asin."
    return _element_unary(known_shapes, node)


def shape_asinh(known_shapes, node):
    "Infers shape for operator Asinh."
    return _element_unary(known_shapes, node)


def shape_atan(known_shapes, node):
    "Infers shape for operator Atan."
    return _element_unary(known_shapes, node)


def shape_atanh(known_shapes, node):
    "Infers shape for operator Atanh."
    return _element_unary(known_shapes, node)


def shape_castlike(known_shapes, node):
    "Infers shape for operator CastLike."
    x = known_shapes[node.input[0]]
    if x.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            f"Result {x!r} must be a tensor.")
    y = known_shapes[node.input[1]]
    if y.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            f"Result {y!r} must be a tensor.")
    cp = x.copy()
    cp.dtype = y.dtype
    return known_shapes.update(node.output[0], cp)


def shape_ceil(known_shapes, node):
    "Infers shape for operator Ceil."
    return _element_unary(known_shapes, node)


def shape_celu(known_shapes, node):
    "Infers shape for operator Celu."
    return _element_unary(known_shapes, node)


def shape_clip(known_shapes, node):
    "Infers shape for operator Clip."
    return _element_unary(known_shapes, node, one_input=False)


def shape_cos(known_shapes, node):
    "Infers shape for operator Cos."
    return _element_unary(known_shapes, node)


def shape_cosh(known_shapes, node):
    "Infers shape for operator Cosh."
    return _element_unary(known_shapes, node)


def shape_elu(known_shapes, node):
    "Infers shape for operator Elu."
    return _element_unary(known_shapes, node)


def shape_erf(known_shapes, node):
    "Infers shape for operator Erf."
    return _element_unary(known_shapes, node)


def shape_exp(known_shapes, node):
    "Infers shape for operator Exp."
    return _element_unary(known_shapes, node)


def shape_floor(known_shapes, node):
    "Infers shape for operator Floor."
    return _element_unary(known_shapes, node)


def shape_hardmax(known_shapes, node):
    "Infers shape for operator Hardmax."
    return _element_unary(known_shapes, node)


def shape_hardsigmoid(known_shapes, node):
    "Infers shape for operator HardSigmoid."
    return _element_unary(known_shapes, node)


def shape_identity(known_shapes, node):
    "Infers shape for operator Identity."
    return _element_unary(known_shapes, node)


def shape_isnan(known_shapes, node):
    "Infers shape for operator IsNan."
    return _element_unary(known_shapes, node, numpy.bool_)


def shape_isinf(known_shapes, node):
    "Infers shape for operator IsInf."
    return _element_unary(known_shapes, node, numpy.bool_)


def shape_leakyrelu(known_shapes, node):
    "Infers shape for operator LeakyRelu."
    return _element_unary(known_shapes, node)


def shape_log(known_shapes, node):
    "Infers shape for operator Log."
    return _element_unary(known_shapes, node)


def shape_logsoftmax(known_shapes, node):
    "Infers shape for operator LogSoftmax."
    return shape_softmax(known_shapes, node)


def shape_neg(known_shapes, node):
    "Infers shape for operator Neg."
    return _element_unary(known_shapes, node)


def shape_not(known_shapes, node):
    "Infers shape for operator Not."
    x = known_shapes[node.input[0]]
    if x.dtype != numpy.bool_:
        raise ShapeInferenceException(
            f"Unexpected input type for operator Not {x.dtype!r} (must be bool).")
    return _element_unary(known_shapes, node)


def shape_reciprocal(known_shapes, node):
    "Infers shape for operator Reciprocal."
    return _element_unary(known_shapes, node)


def shape_relu(known_shapes, node):
    "Infers shape for operator Relu."
    return _element_unary(known_shapes, node)


def shape_round(known_shapes, node):
    "Infers shape for operator Round."
    return _element_unary(known_shapes, node)


def shape_selu(known_shapes, node):
    "Infers shape for operator Selu."
    return _element_unary(known_shapes, node)


def shape_shrink(known_shapes, node):
    "Infers shape for operator Shrink."
    return _element_unary(known_shapes, node)


def shape_sigmoid(known_shapes, node):
    "Infers shape for operator Sigmoid."
    return _element_unary(known_shapes, node)


def shape_sign(known_shapes, node):
    "Infers shape for operator Sigmoid."
    return _element_unary(known_shapes, node)


def shape_sin(known_shapes, node):
    "Infers shape for operator Sin."
    return _element_unary(known_shapes, node)


def shape_sinh(known_shapes, node):
    "Infers shape for operator Sinh."
    return _element_unary(known_shapes, node)


def shape_softmax(known_shapes, node):
    "Infers shape for operator Softmax."
    return _element_unary(known_shapes, node)


def shape_softplus(known_shapes, node):
    "Infers shape for operator Softplus."
    return _element_unary(known_shapes, node)


def shape_softsign(known_shapes, node):
    "Infers shape for operator Softsign."
    return _element_unary(known_shapes, node)


def shape_sqrt(known_shapes, node):
    "Infers shape for operator Sqrt."
    return _element_unary(known_shapes, node)


def shape_tan(known_shapes, node):
    "Infers shape for operator Tan."
    return _element_unary(known_shapes, node)


def shape_tanh(known_shapes, node):
    "Infers shape for operator Tanh."
    return _element_unary(known_shapes, node)


def shape_thresholdedrelu(known_shapes, node):
    "Infers shape for operator ThresholdedRelu."
    return _element_unary(known_shapes, node)


def shape_trilu(known_shapes, node):
    "Infers shape for operator Trilu."
    return _element_unary(known_shapes, node)
