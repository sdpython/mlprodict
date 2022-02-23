
=======================
Export ONNX into Python
=======================

.. contents::
    :local:

Through OnnxInference
=====================

The Python Runtime can be optimized by generating
custom python code and dynamically compile it.
:class:`OnnxInference <mlprodict.onnxrt.onnx_inference.OnnxInference>`
computes predictions based on an ONNX graph with a
python runtime or :epkg:`onnxruntime`.
Method :meth:`to_python
<mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport.to_python>`
goes further by converting the ONNX graph into a standalone
python code. All operators may not be implemented.

External tools
==============

Another tool is implemented in
`onnx2py.py <https://github.com/microsoft/onnxconverter-common/
blob/master/onnxconverter_common/onnx2py.py>`_ and converts an ONNX
graph into a python code which produces this graph.

Export functions
================

The following function converts an ONNX graph into Python code.

onnx API
++++++++

The python code creates the same exported onnx graph with
:epkg:`onnx` API.

.. autosignature:: mlprodict.onnx_tools.onnx_export.export2onnx

to numpy
++++++++

.. index:: numpy

The python code creates a python function using numpy to
produce the same results as the ONNX graph.

.. autosignature:: mlprodict.onnx_tools.onnx_export.export2numpy

tf2onnx
+++++++

.. index:: tf2onnx

This function was used to write a converter for a function
from *tensorflow* (RFFT). To speed up the development, the first
step consisted into writing a numpy function equivalent to the
tensorflow version. Then this function was converted into ONNX
using the numpy API for ONNX. Finally, the ONNX graph was exported
into a python code following tf2onnx API.

.. autosignature:: mlprodict.onnx_tools.onnx_export.export2tf2onnx
