
onnxrt
======

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

ONNX Structure
++++++++++++++

.. autosignature:: mlprodict.onnxrt.onnx_inference_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.onnxrt.onnx_inference_manipulations.select_model_inputs_outputs

Validation
++++++++++

.. autosignature:: mlprodict.onnxrt.validate.validate.enumerate_validated_operator_opsets

.. autosignature:: mlprodict.onnxrt.validate.side_by_side.side_by_side_by_values

.. autosignature:: mlprodict.onnxrt.validate.validate.summary_report

.. autosignature:: mlprodict.onnxrt.model_checker.onnx_shaker

.. autosignature:: mlprodict.onnxrt.validate.validate_graph.plot_validate_benchmark

C++ classes
+++++++++++

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_svm_classifier_.RuntimeSVMClassifier

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_svm_regressor_.RuntimeSVMRegressor

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifierFloat

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifierDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_.RuntimeTreeEnsembleRegressorDouble

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_.RuntimeTreeEnsembleRegressorFloat

Optimisation
++++++++++++

The following functions reduce the number of ONNX operators in a graph
while keeping the same results. The optimized graph
is left unchanged.

.. autosignature:: mlprodict.onnxrt.optim.onnx_optimisation.onnx_remove_node

.. autosignature:: mlprodict.onnxrt.optim.onnx_optimisation_identity.onnx_remove_node_identity

.. autosignature:: mlprodict.onnxrt.optim.onnx_optimisation_redundant.onnx_remove_node_redundant

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
