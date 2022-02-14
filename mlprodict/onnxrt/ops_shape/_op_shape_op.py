"""
@file
@brief Computes shape inference for onnx operators.
"""
from .shape_excs import ShapeInferenceException
from .shape_result import (
    ShapeResult, OnnxKind, ShapeConstraintList, ShapeConstraint)


def shape_det(known_shapes, node):
    "Infers shape for operator Abs."
    x = known_shapes[node.input[0]]
    if x.mtype != OnnxKind.Tensor:
        raise ShapeInferenceException(  # pragma: no cover
            "Result %r must be a tensor." % x)
    if x.n_dims() < 2:
        raise ShapeInferenceException(  # pragma: no cover
            "Operator Det requires at least two dimensions not %r." % x.n_dims())
    name = node.output[0]

    constraints = ShapeConstraintList()
    a, b = x.shape[-2:]
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            raise ShapeInferenceException(  # pragma: no cover
                "Operator Det only applies on square matrices not %r." % x.n_dims())
    elif isinstance(a, str):
        constraints.append(ShapeConstraint(a, {b}))
    elif isinstance(b, str):
        constraints.append(ShapeConstraint(b, {a}))
    else:
        raise ShapeInferenceException(  # pragma: no cover
            "Unexpected case for operator Det (%r)." % x)
    if x.n_dims() == 2:
        r = ShapeResult(name, [], x.dtype, False,
                        x.mtype, constraints)
    else:
        r = ShapeResult(name, x.shape[:-2], x.dtype, False,
                        x.mtype, constraints)
    return known_shapes.update(name, r)
