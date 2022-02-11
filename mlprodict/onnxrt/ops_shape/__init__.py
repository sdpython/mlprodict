"""
@file
@brief Shortcut to *ops_shape*.
"""
from ._element_wise import shape_add, shape_mul, shape_div, shape_sub


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
            node.op_type, node.doomain,
            "\n".join(sorted(_shape_functions))))
