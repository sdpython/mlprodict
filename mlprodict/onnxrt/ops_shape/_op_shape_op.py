"""
@file
@brief Computes shape inference for onnx operators.
"""
from .shape_excs import ShapeInferenceException, ShapeInferenceDimensionError
from .shape_result import (
    ShapeResult, OnnxKind, ShapeConstraintList, ShapeConstraint)


def shape_det(known_shapes, node):
    "Infers shape for operator Abs."
    x = known_shapes[node.input[0]]
    if x.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            f"Result {x!r} must be a tensor.")
    if x.n_dims() < 2:
        if x.n_dims() > 0:
            raise ShapeInferenceException(  # pragma: no cover
                f"Operator Det requires at least two dimensions not {x.n_dims()!r}.")
        raise ShapeInferenceDimensionError(  # pragma: no cover
            f"Operator Det requires at least two dimensions not {x.n_dims()!r}.")
    name = node.output[0]

    constraints = ShapeConstraintList()
    a, b = x.shape[-2:]
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            raise ShapeInferenceException(  # pragma: no cover
                f"Operator Det only applies on square matrices not {x.n_dims()!r}.")
    elif isinstance(a, str):
        constraints.append(ShapeConstraint(a, {b}))
    elif isinstance(b, str):
        constraints.append(ShapeConstraint(b, {a}))
    else:
        raise ShapeInferenceException(  # pragma: no cover
            f"Unexpected case for operator Det ({x!r}).")
    if x.n_dims() == 2:
        r = ShapeResult(name, [], x.dtype, False,
                        x.mtype, constraints)
    else:
        r = ShapeResult(name, x.shape[:-2], x.dtype, False,
                        x.mtype, constraints)
    return known_shapes.update(name, r)
