"""
@file
@brief Computes shape inference for element wise operators.
"""
from .shape_excs import ShapeInferenceException
from .shape_result import ShapeResult, OnnxKind


def _element_wise(known_shapes, node):
    """
    Infers shape for an element wise operator.
    The function returns but updates *known_shapes*.

    :param known_shapes: known shapes
    :param node: Onnx node
    :return: updated or not
    """
    x = known_shapes[node.input[0]]
    y = known_shapes[node.input[1]]
    if x.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            "Result %r must be a tensor." % x)
    if y.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            "Result %r must be a tensor." % y)
    return known_shapes.update(
        node.output[0], ShapeResult.broadcast(x, y, name=node.output[0]))


def shape_add(known_shapes, node):
    "Infers shape for operator Add."
    return _element_wise(known_shapes, node)


def shape_and(known_shapes, node):
    "Infers shape for operator And."
    return _element_wise(known_shapes, node)


def shape_div(known_shapes, node):
    "Infers shape for operator Div."
    return _element_wise(known_shapes, node)


def shape_equal(known_shapes, node):
    "Infers shape for operator Equal."
    return _element_wise(known_shapes, node)


def shape_greater(known_shapes, node):
    "Infers shape for operator Greater."
    return _element_wise(known_shapes, node)


def shape_greaterorequal(known_shapes, node):
    "Infers shape for operator GreaterOrEqual."
    return _element_wise(known_shapes, node)


def shape_less(known_shapes, node):
    "Infers shape for operator Less."
    return _element_wise(known_shapes, node)


def shape_lessorequal(known_shapes, node):
    "Infers shape for operator LessOrEqual."
    return _element_wise(known_shapes, node)


def shape_max(known_shapes, node):
    "Infers shape for operator Max."
    return _element_wise(known_shapes, node)


def shape_min(known_shapes, node):
    "Infers shape for operator Min."
    return _element_wise(known_shapes, node)


def shape_mod(known_shapes, node):
    "Infers shape for operator Mod."
    return _element_wise(known_shapes, node)


def shape_mul(known_shapes, node):
    "Infers shape for operator Mul."
    return _element_wise(known_shapes, node)


def shape_or(known_shapes, node):
    "Infers shape for operator Or."
    return _element_wise(known_shapes, node)


def shape_pow(known_shapes, node):
    "Infers shape for operator Pow."
    return _element_wise(known_shapes, node)


def shape_sub(known_shapes, node):
    "Infers shape for operator Sub."
    return _element_wise(known_shapes, node)


def shape_xor(known_shapes, node):
    "Infers shape for operator Xor."
    return _element_wise(known_shapes, node)
