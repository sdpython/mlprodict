# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnx_conv*.
"""
import onnx
from .register import register_converters, register_scorers
from .register_rewritten_converters import register_rewritten_operators
from .convert import (
    to_onnx, guess_schema_from_data, guess_schema_from_model,
    get_inputs_from_data)
