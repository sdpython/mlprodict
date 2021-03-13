# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
from ._op import OpRunOnnxEmpty


def load_op(onnx_node, desc=None, options=None, variables=None, dtype=None):
    """
    Gets the operator related to the *onnx* node.
    This runtime does nothing and never complains.

    :param onnx_node: :epkg:`onnx` node
    :param desc: internal representation
    :param options: runtime options
    :param variables: registered variables created by previous operators
    :param dtype: float computation type
    :return: runtime class
    """
    if desc is None:
        raise ValueError(  # pragma: no cover
            "desc should not be None.")
    return OpRunOnnxEmpty(onnx_node, desc, variables=variables,
                          dtype=dtype, **options)
