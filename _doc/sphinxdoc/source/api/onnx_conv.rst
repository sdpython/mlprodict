
onnx converters
===============

This packages implements or rewrites some of the
existing converters. They can be registered and uses by
:epkg:`sklearn-onnx` by using the following function:

.. autosignature:: mlprodict.onnx_conv.register.register_converters

.. contents::
    :local:

LightGBM
++++++++

Converters for package :epkg:`lightgbm`.

.. autosignature:: mlprodict.onnx_conv.operator_converters.conv_lightgbm.convert_lightgbm

XGBoost
+++++++

Converters for package :epkg:`xgboost`.

.. autosignature:: mlprodict.onnx_conv.operator_converters.conv_xgboost.convert_xgboost
