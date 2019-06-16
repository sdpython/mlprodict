
.. _l-onnx-runtime-operators:

Python Runtime for ONNX operators
=================================

The main function instantiates a runtime class which
computes the outputs of a specific node.

.. autosignature:: mlprodict.onnxrt.ops.load_op

Other sections documents available operators.
This project was mostly started to show a way to
implement a custom runtime, do some benchmarks,
test, exepriment...

.. contents::
    :local:

CPU
+++

Add
^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_add.Add

ArgMax
^^^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_argmax.ArgMax

ArgMin
^^^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_argmin.ArgMin

Cast
^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_cast.Cast

Div
^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_div.Div

Gemm
^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_gemm.Gemm

LinearClassifier
^^^^^^^^^^^^^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_linear_classifier.LinearClassifier

LinearRegressor
^^^^^^^^^^^^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_linear_regressor.LinearRegressor

Normalizer
^^^^^^^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_normalizer.Normalizer

ZipMap
^^^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_zipmap.ZipMap

OnnxRuntime
+++++++++++

.. autosignature:: mlprodict.onnxrt.ops_onnxruntime.load_op

.. autosignature:: mlprodict.onnxrt.ops_onnxruntime._op.OpRunOnnxRuntime
