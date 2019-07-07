
onnxrt
======

.. content::
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

.. autosignature:: mlprodict.onnx_grammar import translate_fct2onnx

Structure
+++++++++

.. autosignature:: mlprodict.onnxrt.onnx_inference_manipulations.enumerate_model_node_outputs

.. autosignature:: mlprodict.onnxrt.onnx_inference_manipulations.select_model_inputs_outputs

Validation
++++++++++

.. autosignature:: mlprodict.onnxrt.validation.enumerate_validated_operator_opsets

.. autosignature:: mlprodict.onnxrt.side_by_side.side_by_side_by_values

.. autosignature:: mlprodict.onnxrt.validation.summary_report
