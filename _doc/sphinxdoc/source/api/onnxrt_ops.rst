
.. _l-onnx-runtime-operators:

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

Add
^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_add.Add

Cast
^^^^

.. autosignature:: mlprodict.onnxrt.ops_cpu.op_cast.Cast

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
