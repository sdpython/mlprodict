
Python Runtime for ONNX operators
=================================

The main function instantiates a runtime class which
computes the outputs of a specific node.

.. autosignature:: mlprodict.onnxrt.ops.load_ops

Other sections documents available operators.
This project was mostly started to show a way to
implement a custom runtime, do some benchmarks,
test, exepriment...

.. contents::
    :local:

CPU
+++

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_add.Add

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_cast.Cast

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_linear_classifier.LinearClassifier

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_linear_regressor.LinearRegressor

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_normalizer.Normalizer

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_zipmap.ZipMap
