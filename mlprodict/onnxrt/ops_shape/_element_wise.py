"""
@file
@brief Computes shape inference for element wise operators.
"""
from .shape_result import ShapeResult


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
    return known_shapes.update(
        node.output[0], ShapeResult.broadcast(x, y, name=node.output[0]))


def shape_add(known_shapes, node):
    "Infers shape for operator Add."
    return _element_wise(known_shapes, node)


def shape_sub(known_shapes, node):
    "Infers shape for operator Sub."
    return _element_wise(known_shapes, node)


def shape_div(known_shapes, node):
    "Infers shape for operator Div."
    return _element_wise(known_shapes, node)


def shape_mul(known_shapes, node):
    "Infers shape for operator Mul."
    return _element_wise(known_shapes, node)
