
onnx converters
===============

This packages implements or rewrites some of the
existing converters. They can be registered and uses by
:epkg:`sklearn-onnx` by using the following function:

.. autosignature:: mlprodict.onnx_conv.register.register_converters

.. autosignature:: mlprodict.onnx_conv.register_rewritten_converters.register_rewritten_operators

.. contents::
    :local:

LightGBM
++++++++

Converters for package :epkg:`lightgbm`.

.. autosignature:: mlprodict.onnx_conv.operator_converters.conv_lightgbm.convert_lightgbm

scikit-learn
++++++++++++

A couple of custom converters were written to test
scenarios not necessarily part of the official ONNX
specifications.

.. autosignature:: mlprodict.onnx_conv.sklconv.ada_boost.convert_sklearn_ada_boost_regressor

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.convert_sklearn_decision_tree_regressor

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.convert_sklearn_gradient_boosting_regressor

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.convert_sklearn_random_forest_regressor_converter

.. autosignature:: mlprodict.onnx_conv.sklconv.svm_converters.convert_sklearn_svm

XGBoost
+++++++

Converters for package :epkg:`xgboost`.

.. autosignature:: mlprodict.onnx_conv.operator_converters.conv_xgboost.convert_xgboost
