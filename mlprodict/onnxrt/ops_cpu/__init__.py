# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
from onnx.defs import onnx_opset_version
from ._op_list import __dict__ as d_op_list


def get_opset_number_from_onnx():
    """
    Retuns the current :epkg:`onnx` opset
    based on the installed version of :epkg:`onnx`.
    """
    return onnx_opset_version()


def load_op(onnx_node, desc=None, options=None):
    """
    Gets the operator related to the *onnx* node.

    @param      onnx_node       :epkg:`onnx` node
    @param      desc            internal representation
    @param      options         runtime options
    @return                     runtime class
    """
    if desc is None:
        raise ValueError("desc should not be None.")
    name = onnx_node.op_type
    opset = options.get('target_opset', None) if options is not None else None
    current_opset = get_opset_number_from_onnx()
    chosen_opset = current_opset
    if opset == current_opset:
        opset = None
    if opset is not None:
        if not isinstance(opset, int):
            raise TypeError(
                "opset must be an integer not {}".format(type(opset)))
        name_opset = name + "_" + str(opset)
        for op in range(opset, 0, -1):
            nop = name + "_" + str(op)
            if nop in d_op_list:
                name_opset = nop
                chosen_opset = op
                break
    else:
        name_opset = name

    if name_opset in d_op_list:
        cl = d_op_list[name_opset]
    elif name in d_op_list:
        cl = d_op_list[name]
    else:
        raise NotImplementedError("Operator '{}' has no runtime yet. Available list:\n"
                                  "{}".format(name, "\n".join(sorted(d_op_list))))

    if hasattr(cl, 'version_higher_than'):
        opv = min(current_opset, chosen_opset)
        if cl.version_higher_than > opv:
            # The chosen implementation does not support
            # the opset version, we need to downgrade it.
            if 'target_opset' in options and options['target_opset'] is not None:
                raise RuntimeError(
                    "Unable to find an implementation version {} found "
                    "'{}'\n--ONNX--\n{}\n--AVAILABLE--\n{}".format(
                        options['target_opset'], cl.__name__, onnx_node,
                        "\n".join(sorted(d_op_list))))
            options = options.copy()
            options['target_opset'] = current_opset
            return load_op(onnx_node, desc=desc, options=options)

    if options is None:
        options = {}
    return cl(onnx_node, desc=desc, **options)
