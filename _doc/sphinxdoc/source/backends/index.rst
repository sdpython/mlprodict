
ONNX Backends
=============

:epkg:`onnx` package implements a series of tests telling how many
operators and cases are supported by a runtime. These tests
are available through an API: :epkg:`ONNX Backend`.
This API was implemented for class :class:`OnnxInference
<mlprodict.onnxrt.onnx_inference.OnnxInference>` and runtimes
`python` and `onnxruntime1` through class :class:`OnnxInferenceBackend
<mlprodict.onnxrt.backend.OnnxInferenceBackend>` and
:class:`OnnxInferenceBackendRep
<mlprodict.onnxrt.backend.OnnxInferenceBackendRep>`.
Following pages share a code example to run this back on all short
tests.

.. toctree::
    :maxdepth: 1

    backend_python
    backend_onnxruntime1
