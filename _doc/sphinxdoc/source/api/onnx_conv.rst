
Additional ONNX Converters
==========================

This packages implements or rewrites some of the
existing converters. They can be registered and uses by
:epkg:`sklearn-onnx` by using the following function:

.. autosignature:: mlprodict.onnx_conv.register.register_converters

.. autosignature:: mlprodict.onnx_conv.register_rewritten_converters.register_rewritten_operators

.. autosignature:: mlprodict.onnx_conv.convert.to_onnx

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

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.new_convert_sklearn_decision_tree_classifier

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.new_convert_sklearn_decision_tree_regressor

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.new_convert_sklearn_gradient_boosting_classifier

.. autosignature:: mlprodict.onnx_conv.sklconv.tree_converters.new_convert_sklearn_gradient_boosting_regressor

.. autosignature:: mlprodict.onnx_conv.sklconv.svm_converters.new_convert_sklearn_random_forest_classifier

.. autosignature:: mlprodict.onnx_conv.sklconv.svm_converters.new_convert_sklearn_random_forest_regressor

SVM
+++

.. autosignature:: mlprodict.onnx_conv.sklconv.svm_converters.new_convert_sklearn_svm_classifier

.. autosignature:: mlprodict.onnx_conv.sklconv.svm_converters.new_convert_sklearn_svm_regressor

XGBoost
+++++++

Converters for package :epkg:`xgboost`.

.. autosignature:: mlprodict.onnx_conv.operator_converters.conv_xgboost.convert_xgboost
