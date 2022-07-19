"""
@file
@brief Computes shape inference for element wise operators.
"""
import numpy
from .shape_excs import ShapeInferenceException
from .shape_result import ShapeResult, OnnxKind


def _element_wise(known_shapes, node, return_bool=False, same_type=True,
                  one_input=False):
    """
    Infers shape for an element wise operator.
    The function returns but updates *known_shapes*.

    :param known_shapes: known shapes
    :param node: Onnx node
    :param return_bool: return boolean
    :param same_type: check the type are the same
    :param one_input: allow one input
    :return: updated or not
    """
    if one_input:
        if len(node.input) == 1:
            x = known_shapes[node.input[0]]
            return known_shapes.update(node.output[0], x.copy())
    elif len(node.input) != 2:
        raise ShapeInferenceException(  # pragma: no cover
            f"Node {node.name!r} must have two inputs not {len(node.input)}.")
    x = known_shapes[node.input[0]]
    y = known_shapes[node.input[1]]
    if x.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            f"Result {x!r} must be a tensor.")
    if y.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            f"Result {y!r} must be a tensor.")
    if return_bool:
        return known_shapes.update(
            node.output[0],
            ShapeResult.broadcast(
                x, y, name=node.output[0], dtype=numpy.bool_,
                same_type=same_type))
    return known_shapes.update(
        node.output[0],
        ShapeResult.broadcast(
            x, y, name=node.output[0], same_type=same_type))


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
    return _element_wise(known_shapes, node, return_bool=True)


def shape_greater(known_shapes, node):
    "Infers shape for operator Greater."
    return _element_wise(known_shapes, node, return_bool=True)


def shape_greaterorequal(known_shapes, node):
    "Infers shape for operator GreaterOrEqual."
    return _element_wise(known_shapes, node, return_bool=True)


def shape_less(known_shapes, node):
    "Infers shape for operator Less."
    return _element_wise(known_shapes, node, return_bool=True)


def shape_lessorequal(known_shapes, node):
    "Infers shape for operator LessOrEqual."
    return _element_wise(known_shapes, node, return_bool=True)


def shape_max(known_shapes, node):
    "Infers shape for operator Max."
    return _element_wise(known_shapes, node, one_input=True)


def shape_min(known_shapes, node):
    "Infers shape for operator Min."
    return _element_wise(known_shapes, node, one_input=True)


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
    return _element_wise(known_shapes, node, same_type=False)


def shape_sub(known_shapes, node):
    "Infers shape for operator Sub."
    return _element_wise(known_shapes, node)


def shape_xor(known_shapes, node):
    "Infers shape for operator Xor."
    return _element_wise(known_shapes, node)
