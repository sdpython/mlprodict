
onnxrt
======

.. contents::
    :local:

Inference
+++++++++

The main class reads an :epkg:`ONNX` file
and may computes predictions based on a runtime
implementated in Python. The ONNX model relies
on the following operators :ref:`l-onnx-runtime-operators`.

.. autosignature:: mlprodict.onnxrt.onnx_inference.OnnxInference
    :members:

Python to ONNX
++++++++++++++

.. autosignature:: mlprodict.onnx_grammar.onnx_translation.translate_fct2onnx

Structure
+++++++++

.. autosignature:: mlprodict.onnxrt.onnx_inference_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.onnxrt.onnx_inference_manipulations.select_model_inputs_outputs

Validation
++++++++++

.. autosignature:: mlprodict.onnxrt.validate.enumerate_validated_operator_opsets

.. autosignature:: mlprodict.onnxrt.side_by_side.side_by_side_by_values

.. autosignature:: mlprodict.onnxrt.validate.summary_report

.. autosignature:: mlprodict.onnxrt.model_checker.onnx_shaker

C++ classes
+++++++++++

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_svm_classifier_.RuntimeSVMClassifier

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_svm_regressor_.RuntimeSVMRegressor

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifier

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_.RuntimeTreeEnsembleRegressor
