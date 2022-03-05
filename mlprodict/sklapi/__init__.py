# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *sklapi*.
Importing this file imports :epkg:`sklearn-onnx` as well.
"""
from .onnx_pipeline import OnnxPipeline
from .onnx_transformer import OnnxTransformer
from .onnx_speed_up import (
    OnnxSpeedupClassifier,
    OnnxSpeedupCluster,
    OnnxSpeedupRegressor,
    OnnxSpeedupTransformer)
