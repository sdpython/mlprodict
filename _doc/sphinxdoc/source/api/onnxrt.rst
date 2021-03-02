
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
    :members:

Python to ONNX
++++++++++++++

.. autosignature:: mlprodict.onnx_grammar.onnx_translation.translate_fct2onnx

ONNX Export
+++++++++++

.. autosignature:: mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport

ONNX Structure
++++++++++++++

.. autosignature:: mlprodict.tools.onnx_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.tools.onnx_manipulations.select_model_inputs_outputs

Validation
++++++++++

.. autosignature:: mlprodict.onnxrt.validate.validate.enumerate_validated_operator_opsets

.. autosignature:: mlprodict.onnxrt.validate.side_by_side.side_by_side_by_values

.. autosignature:: mlprodict.onnxrt.validate.validate_summary.summary_report

.. autosignature:: mlprodict.onnxrt.model_checker.onnx_shaker

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

Optimisation
++++++++++++

The following functions reduce the number of ONNX operators in a graph
while keeping the same results. The optimized graph
is left unchanged.

.. autosignature:: mlprodict.onnxrt.optim.onnx_optimisation.onnx_remove_node

.. autosignature:: mlprodict.onnxrt.optim.onnx_optimisation_identity.onnx_remove_node_identity

.. autosignature:: mlprodict.onnxrt.optim.onnx_optimisation_redundant.onnx_remove_node_redundant

.. autosignature:: mlprodict.onnxrt.optim.onnx_remove_unused.onnx_remove_node_unused

Shapes
++++++

The computation of the predictions through epkg:`ONNX` may
be optimized if the shape of every nodes is known. For example,
one possible optimisation is to do inplace computation every time
it is possible but this is only possible if the size of
the input and output are the same. We could compute the predictions
for a sample and check the sizes are the same
but that could be luck. We could also guess from a couple of samples
with different sizes and assume sizes and polynomial functions
of the input size. But in rare occasions, that could be luck too.
So one way of doing it is to implement a method
:meth:`_set_shape_inference_runtime
<mlprodict.onnxrt.onnx_inference.OnnxInference._set_shape_inference_runtime>`
which works the same say as method :meth:`_run_sequence_runtime
<mlprodict.onnxrt.onnx_inference.OnnxInference._run_sequence_runtime>`
but handles shapes instead. Following class tries to implement
a way to keep track of shape along the shape.

.. autosignature:: mlprodict.onnxrt.shape_object.ShapeObject
