
.. _l-numpy-onnxpy:

Complete Numpy API for ONNX
===========================

The numpy API is meant to simplofy the creation of ONNX
graphs by using functions very similar to what numpy implements.
This page only makes a list of the available
functions. A tutorial is available at
:ref:`l-numpy-api-for-onnx`.
This API was first added to *mlprodict* in version 0.6.

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
The decorator returns a function which is strict about types
as opposed to numpy. This approach is similar to what
:epkg:`tensorflow` with `autograph
<https://www.tensorflow.org/api_docs/python/tf/autograph>`_.

Signatures
++++++++++

.. autosignature:: mlprodict.npy.onnx_numpy_annotation.NDArray
    :members:

.. autosignature:: mlprodict.npy.onnx_numpy_annotation.NDArraySameType
    :members:

.. autosignature:: mlprodict.npy.onnx_numpy_annotation.NDArraySameTypeSameShape
    :members:

.. autosignature:: mlprodict.npy.onnx_numpy_annotation.NDArrayType
    :members:

.. autosignature:: mlprodict.npy.onnx_numpy_annotation.NDArrayTypeSameShape
    :members:

Decorators
++++++++++

.. autosignature:: mlprodict.npy.onnx_numpy_wrapper.onnxnumpy

.. autosignature:: mlprodict.npy.onnx_numpy_wrapper.onnxnumpy_default

.. autosignature:: mlprodict.npy.onnx_numpy_wrapper.onnxnumpy_np

.. autosignature:: mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_class

.. autosignature:: mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_classifier

.. autosignature:: mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_cluster

.. autosignature:: mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_regressor

.. autosignature:: mlprodict.npy.onnx_sklearn_wrapper.onnxsklearn_transformer

OnnxNumpyCompiler
+++++++++++++++++

.. autosignature:: mlprodict.npy.onnx_numpy_compiler.OnnxNumpyCompiler
    :members:

.. autosignature::  mlprodict.npy.onnx_version.FctVersion
    :members:

OnnxVar
+++++++

.. autosignature:: mlprodict.npy.onnx_variable.OnnxVar
    :members:

.. autosignature:: mlprodict.npy.onnx_variable.MultiOnnxVar
    :members:

.. autosignature:: mlprodict.npy.onnx_variable.TupleOnnxAny
    :members:

Registration
++++++++++++

.. autosignature:: mlprodict.npy.onnx_sklearn_wrapper.update_registered_converter_npy

.. _l-numpy-onnxpy-list-fct:

Available functions implemented with ONNX operators
+++++++++++++++++++++++++++++++++++++++++++++++++++

All functions are implemented in two submodules:

* *numpy function*: :ref:`f-numpyonnximpl`
* *machine learned models:* :ref:`f-numpyonnximplskl`

ONNX functions executed python ONNX runtime
+++++++++++++++++++++++++++++++++++++++++++

Same function as above, the import goes from
``from mlprodict.npy.numpy_onnx_impl import <function-name>`` to
``from mlprodict.npy.numpy_onnx_pyrt import <function-name>``.
These function are usually not used except in unit test or as
reference for more complex functions. See the source on github,
`numpy_onnx_pyrt.py
<https://github.com/sdpython/mlprodict/blob/master/mlprodict/npy/numpy_onnx_pyrt.py>`_
and `numpy_onnx_pyrt_skl.py
<https://github.com/sdpython/mlprodict/blob/master/mlprodict/npy/numpy_onnx_pyrt_skl.py>`_.
