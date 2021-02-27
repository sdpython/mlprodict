
.. _l-numpy-onnxpy:

Numpy API for ONNX
==================

.. contents::
    :local:

Introduction
++++++++++++

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
    :warningout: DeprecationWarning

    import numpy
    from typing import Any
    from mlprodict.npy import onnxnumpy_default, NDArray
    import mlprodict.npy.numpy_onnx_impl as nxnp

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

NDArray
+++++++

.. autosignature:: mlprodict.npy.onnx_numpy_annotation.NDArray
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

Available numpy functions implemented with ONNX operators
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autosignature:: mlprodict.npy.numpy_onnx_impl.abs

.. autosignature:: mlprodict.npy.numpy_onnx_impl.acos

.. autosignature:: mlprodict.npy.numpy_onnx_impl.acosh

.. autosignature:: mlprodict.npy.numpy_onnx_impl.amax

.. autosignature:: mlprodict.npy.numpy_onnx_impl.amin

.. autosignature:: mlprodict.npy.numpy_onnx_impl.argmax

.. autosignature:: mlprodict.npy.numpy_onnx_impl.argmin

.. autosignature:: mlprodict.npy.numpy_onnx_impl.asin

.. autosignature:: mlprodict.npy.numpy_onnx_impl.asinh

.. autosignature:: mlprodict.npy.numpy_onnx_impl.atan

.. autosignature:: mlprodict.npy.numpy_onnx_impl.atanh

.. autosignature:: mlprodict.npy.numpy_onnx_impl.ceil

.. autosignature:: mlprodict.npy.numpy_onnx_impl.cos

.. autosignature:: mlprodict.npy.numpy_onnx_impl.cosh

.. autosignature:: mlprodict.npy.numpy_onnx_impl.erf

.. autosignature:: mlprodict.npy.numpy_onnx_impl.exp

.. autosignature:: mlprodict.npy.numpy_onnx_impl.isnan

.. autosignature:: mlprodict.npy.numpy_onnx_impl.mean

.. autosignature:: mlprodict.npy.numpy_onnx_impl.log

.. autosignature:: mlprodict.npy.numpy_onnx_impl.prod

.. autosignature:: mlprodict.npy.numpy_onnx_impl.reciprocal

.. autosignature:: mlprodict.npy.numpy_onnx_impl.relu

.. autosignature:: mlprodict.npy.numpy_onnx_impl.round

.. autosignature:: mlprodict.npy.numpy_onnx_impl.sign

.. autosignature:: mlprodict.npy.numpy_onnx_impl.sin

.. autosignature:: mlprodict.npy.numpy_onnx_impl.sinh

.. autosignature:: mlprodict.npy.numpy_onnx_impl.sqrt

.. autosignature:: mlprodict.npy.numpy_onnx_impl.sum

.. autosignature:: mlprodict.npy.numpy_onnx_impl.tan

.. autosignature:: mlprodict.npy.numpy_onnx_impl.tanh

ONNX functions executed python ONNX runtime
+++++++++++++++++++++++++++++++++++++++++++

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.abs

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.acos

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.acosh

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.amax

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.amin

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.argmax

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.argmin

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.asin

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.asinh

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.atan

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.atanh

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.ceil

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.cos

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.cosh

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.erf

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.exp

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.isnan

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.log

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.mean

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.prod

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.reciprocal

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.relu

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.round

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.sign

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.sin

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.sinh

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.sqrt

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.sum

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.tan

.. autosignature:: mlprodict.npy.numpy_onnx_pyrt.tanh
