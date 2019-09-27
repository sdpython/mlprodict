# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnx_conv*.
"""
from .register import register_converters, register_scorers
from .convert import to_onnx, guess_schema_from_data, guess_schema_from_model
