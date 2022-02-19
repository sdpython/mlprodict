"""
@file
@brief Shortcut to *ops_shape*.
"""
from ._element_unary import (
    shape_abs, shape_acos, shape_acosh,
    shape_asin, shape_asinh, shape_atan, shape_atanh,
    shape_castlike, shape_ceil, shape_celu,
    shape_clip, shape_cos, shape_cosh,
    shape_erf, shape_exp, shape_floor, shape_identity, shape_isnan,
    shape_leakyrelu, shape_log,
    shape_neg, shape_not, shape_reciprocal, shape_relu, shape_round,
    shape_sigmoid, shape_sign, shape_sin, shape_sinh, shape_softmax,
    shape_sqrt, shape_tan, shape_tanh)
from ._element_wise import (
    shape_add, shape_and,
    shape_div,
    shape_equal,
    shape_greater, shape_greaterorequal,
    shape_less, shape_lessorequal,
    shape_max, shape_min, shape_mod, shape_mul,
    shape_or,
    shape_pow,
    shape_sub)
from ._op_shape_op import shape_det


_shape_functions = {
    k: v for k, v in globals().items() if k.startswith("shape_")
}


def shape_dispatch(known_shape, node):
    """
    Calls the corresponding fucntion for every node.

    :param known_shape: known_shape for all results
    :param node: onnx node
    :return: was *known_shape* updated or not...
    """
    op_type = "shape_" + node.op_type.lower()
    if op_type in _shape_functions:
        return _shape_functions[op_type](known_shape, node)
    raise RuntimeError(  # pragma: no cover
        "Unable to find a corresponding function for operator type %r "
        "domain=%r among\n%s" % (
            node.op_type, node.domain,
            "\n".join(sorted(_shape_functions))))
