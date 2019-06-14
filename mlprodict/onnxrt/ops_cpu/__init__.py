# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""

from ._op_list import __dict__ as d_op_list


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
    if name in d_op_list:
        cl = d_op_list[name]
        if options is None:
            options = {}
        return cl(onnx_node, desc=desc, **options)

    raise NotImplementedError("Operator '{}' has no runtime yet.".format(name))
