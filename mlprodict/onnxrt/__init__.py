# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *onnxrt*.
"""
from .onnx_inference import OnnxInference
from .validate.validate_difference import measure_relative_difference
from .validate.validate_helper import sklearn_operators
