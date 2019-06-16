# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
from ._op import OpRunOnnxRuntime


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
    return OpRunOnnxRuntime(onnx_node, desc, **options)
