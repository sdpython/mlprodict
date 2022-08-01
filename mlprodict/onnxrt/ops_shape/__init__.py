"""
@file
@brief Shortcut to *ops_shape*.
"""
import textwrap
from onnx.onnx_cpp2py_export.defs import SchemaError  # pylint: disable=E0401,E0611
from ...onnx_tools.onnx2py_helper import get_onnx_schema
from .shape_excs import ShapeInferenceMissing
from ._element_unary import (
    shape_abs, shape_acos, shape_acosh,
    shape_asin, shape_asinh, shape_atan, shape_atanh,
    shape_castlike, shape_ceil, shape_celu,
    shape_clip, shape_cos, shape_cosh,
    shape_elu, shape_erf, shape_exp, shape_floor,
    shape_hardmax, shape_hardsigmoid,
    shape_identity, shape_isinf, shape_isnan,
    shape_leakyrelu, shape_log, shape_logsoftmax,
    shape_neg, shape_not, shape_reciprocal, shape_relu, shape_round,
    shape_selu, shape_shrink,
    shape_sigmoid, shape_sign, shape_sin, shape_sinh, shape_softmax,
    shape_softplus, shape_softsign,
    shape_sqrt, shape_tan, shape_tanh, shape_thresholdedrelu,
    shape_trilu)
from ._element_wise import (
    shape_add, shape_and,
    shape_div,
    shape_equal,
    shape_greater, shape_greaterorequal,
    shape_less, shape_lessorequal,
    shape_max, shape_min, shape_mod, shape_mul,
    shape_or,
    shape_pow,
    shape_sub,
    shape_xor)
from ._op_shape_op import shape_det


_shape_functions = {
    k: v for k, v in globals().items() if k.startswith("shape_")
}


count = [0]


def shape_dispatch(cache, known_shape, node, rt_class=None):
    """
    Calls the corresponding fucntion for every node.

    :param cache: cache used function
    :param known_shape: known_shape for all results
    :param node: onnx node
    :param rt_class: a node may be a predefined function in onnx,
        if no specific function is available, the predefined
        onnx definition is used and run through this runtime
    :return: was *known_shape* updated or not...
    """
    key = node.domain, node.op_type
    fct_shape = None
    if key in cache:
        fct_shape = cache[key]
    else:
        op_type = "shape_" + node.op_type.lower()
        if op_type in _shape_functions:
            fct_shape = _shape_functions[op_type]
            cache[key] = fct_shape

    if fct_shape is None and rt_class is not None:
        # check this operator is a predefined function in ONNX.
        try:
            onnx_schema = get_onnx_schema(node.op_type, node.domain)
        except SchemaError:
            onnx_schema = None
        if onnx_schema is not None and onnx_schema.has_function:
            sess = rt_class(onnx_schema.function_body)
            if len(node.input) != len(sess.input_names):
                raise RuntimeError(  # pragma: no cover
                    "node and function must have the same number of inputs, "
                    "len(%r) != len(%r)." % (
                        node.input, sess.input_names))
            if len(node.output) != len(sess.output_names):
                raise RuntimeError(  # pragma: no cover
                    "node and function must have the same number of outputs, "
                    "len(%r) != len(%r)." % (
                        node.output, sess.output_names))

            def _shape_function(known_shape, node):
                inputs = {iname: known_shape[name] for name, iname in
                          zip(node.input, sess.input_names)}
                outputs = sess.run(inputs)
                res = False
                for name, oname in zip(node.output, sess.output_names):
                    r = known_shape.update(name, outputs[oname])
                    res = res or r
                return res

            fct_shape = _shape_function
            cache[key] = fct_shape

    if fct_shape is not None:
        return fct_shape(known_shape, node)

    raise ShapeInferenceMissing(  # pragma: no cover
        "Unable to find a corresponding function for operator type %r "
        "domain=%r, looking for %r among\n%s" % (
            node.op_type, node.domain, "shape_" + node.op_type.lower(),
            "\n".join(textwrap.wrap(
                " ".join(_ for _ in sorted(_shape_functions))))))
