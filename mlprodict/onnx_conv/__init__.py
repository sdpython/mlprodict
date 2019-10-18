# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnx_conv*.
"""
import onnx
from .register import register_converters, register_scorers
from .register_rewritten_converters import register_rewritten_operators
from .convert import to_onnx, guess_schema_from_data, guess_schema_from_model


def get_onnx_opset():
    """
    Retuns the current :epkg:`onnx` opset
    based on the installed version of :epkg:`onnx`.
    """
    return onnx.defs.onnx_opset_version()
