
Python Runtime for ONNX
=======================

This runtime does not take any dependency on :epkg:`scikit-learn`,
only on :epkg:`numpy`, :epkg:`scipy`, and has custom implementations
in C++ (:epkg:`cython`, :epkg:`pybind11`).

.. contents::
    :local:

Inference
+++++++++

The main class reads an :epkg:`ONNX` file
and may computes predictions based on a runtime
implementated in :epkg:`Python`. The :epkg:`ONNX` model relies
on the following operators :ref:`l-onnx-runtime-operators`.

.. autosignature:: mlprodict.onnxrt.onnx_inference.OnnxInference
    :members: run, shape_inference, check_model, run2onnx, get_profiling

.. autosignature:: mlprodict.onnxrt.onnx_micro_inference.OnnxMicroRuntime
    :members: run

The following is technically implemented as a runtime but it does
shape inference.

.. autosignature:: mlprodict.onnxrt.onnx_shape_inference.OnnxShapeInference
    :members: run

The execution produces a result of type:

.. autosignature:: mlprodict.onnxrt.ops_shape.shape_container.ShapeContainer
    :members: get

Methods `get` returns a dictionary mapping result name and the following type:

.. autosignature:: mlprodict.onnxrt.ops_shape.shape_result.ShapeResult
    :members:

Backend validation
++++++++++++++++++

.. autosignature:: mlprodict.tools.onnx_backend.enumerate_onnx_tests

.. autosignature:: mlprodict.tools.onnx_backend.OnnxBackendTest

Python to ONNX
++++++++++++++

.. autosignature:: mlprodict.onnx_tools.onnx_grammar.onnx_translation.translate_fct2onnx

ONNX Export
+++++++++++

.. autosignature:: mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport

ONNX Structure
++++++++++++++

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.onnx_tools.onnx_manipulations.select_model_inputs_outputs

onnxruntime
+++++++++++

.. autosignature:: mlprodict.onnxrt.onnx_inference_ort.device_to_providers

.. autosignature:: mlprodict.onnxrt.onnx_inference_ort.get_ort_device

Validation of scikit-learn models
+++++++++++++++++++++++++++++++++

.. autosignature:: mlprodict.onnxrt.validate.validate.enumerate_validated_operator_opsets

.. autosignature:: mlprodict.onnxrt.validate.side_by_side.side_by_side_by_values

.. autosignature:: mlprodict.onnxrt.validate.validate_summary.summary_report

.. autosignature:: mlprodict.onnxrt.validate.validate_graph.plot_validate_benchmark

C++ classes
+++++++++++

**Gather**

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_gather_.GatherDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_gather_.GatherFloat

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_gather_.GatherInt64

**ArrayFeatureExtractor**

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.array_feature_extractor_double

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.array_feature_extractor_float

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.array_feature_extractor_int64

**SVM**

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_svm_classifier_.RuntimeSVMClassifier

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_svm_regressor_.RuntimeSVMRegressor

**Tree Ensemble**

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifierDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifierFloat

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_.RuntimeTreeEnsembleRegressorDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_.RuntimeTreeEnsembleRegressorFloat

**Still tree ensembles but refactored.**

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_p_.RuntimeTreeEnsembleClassifierPDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_p_.RuntimeTreeEnsembleClassifierPFloat

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_p_.RuntimeTreeEnsembleRegressorPDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_p_.RuntimeTreeEnsembleRegressorPFloat

**Topk**

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_max_double

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_max_float

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_max_int64

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_min_double

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_min_float

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_min_int64

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_fetch_double

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_fetch_float

.. autosignature:: mlprodict.onnxrt.ops_cpu._op_onnx_numpy.topk_element_fetch_int64
