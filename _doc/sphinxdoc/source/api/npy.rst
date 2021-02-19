
.. _l-numpy-onnxpy:

Numpy revisited with ONNX
=========================

Converting custom code into :epkg:`ONNX` is not necessarily easy.
One big obstacle is :epkg:`ONNX` does not represent all numpy functions
with a single operator. One possible option is to provide a
:epkg:`numpy` API to :epkg:`ONNX`. That's the purpose of wrapper
:class:`onnxnumpy <mlprodict.npy.onnx_numpy_wrapper.onnxnumpy>`.
It takes a function written with functions following the same
signature as :epkg:`numpy` and provides a way to execute them
with an :epkg:`ONNX` runtime. In the below example,
`custom_fct` creates an :epkg:`ONNX` graph, the wrapper
loads it in a runtime and runs it everytime the function
is called.

.. runpython::
    :showcode:

    import numpy
    from typing import Any
    from mlprodict.npy import onnxnumpy_default, NDArray
    import mlprodict.npy.numpy_impl as nxnp

    @onnxnumpy_default
    def custom_fct(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
        "onnx numpy abs"
        return nxnp.abs(x) + numpy.float32(1)

    x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
    y = custom_fct(x)
    print(y)

Annotations are mandatory to indicate inputs and outputs type.
As a result, the returned function is strict about types
as opposed to numpy. This approach is similar to what
:epkg:`tensorflow` with `autograph
<https://www.tensorflow.org/api_docs/python/tf/autograph>`_.

.. contents::
    :local:

NDArray
+++++++

.. autosignature:: mlprodict.npy.onnx_numpy_compiler.NDArray
    :members:

onnxnumpy
+++++++++

.. autosignature:: mlprodict.npy.onnx_numpy_wrapper.onnxnumpy

.. autosignature:: mlprodict.npy.onnx_numpy_wrapper.onnxnumpy_default

OnnxNumpyCompiler
+++++++++++++++++

.. autosignature:: mlprodict.npy.onnx_numpy_compiler.OnnxNumpyCompiler
    :members:

OnnxVar
+++++++

.. autosignature:: mlprodict.npy.onnx_variable.OnnxVar
    :members:

Available numpy functions
+++++++++++++++++++++++++

.. autosignature:: mlprodict.npy.numpy_impl.abs

.. autosignature:: mlprodict.npy.numpy_impl.sum
