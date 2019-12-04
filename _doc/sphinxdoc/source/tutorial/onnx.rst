
.. _l-onnx-tutorial:

ONNX and python
===============

This package implements a python runtime for ONNX
in class :class:`OnnxInference <mlprodict.onnxrt.OnnxInference>`.
It does not depend on :epkg:`scikit-learn`, only :epkg:`numpy`
and this module. However, this module was not really developped to
get the fastest python runtime but mostly to easily develop converters.
That's why the module can also convert an ONNX graph
into a python code which relies on :epkg:`numpy` in most
case, :epkg:`scipy` and this module in all cases.

.. contents::
    :local:

From ONNX to Python
+++++++++++++++++++

:class:`OnnxInference <mlprodict.onnxrt.OnnxInference>`
was compute predictions based on an ONNX graph with a
python runtime or :epkg:`onnxruntime`.
Method :meth:`to_python
<mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport.to_python>`
goes further by converting the ONNX graph into a standalone
python code.

.. _l-numpy2onnx-tutorial:

From numpy to ONNX
++++++++++++++++++

.. index:: algebric function

*mlprodict* implements function
:func:`translate_fct2onnx
<mlprodict.onnx_grammar.onnx_translation.translate_fct2onnx>`
which converts the code
of a function written with :epkg:`numpy` and :epkg:`scipy`
into an :epkg:`ONNX` graph.

The kernel *ExpSineSquared*
is used by :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`
and its conversion is required to convert the model.
The first step is to write a standalone function which
relies on :epkg:`scipy` or :epkg:`numpy` and which produces
the same results. The second step calls this function to
produces the :epkg:`ONNX` graph.

.. runpython::
    :showcode:
    :process:
    :store_in_file: fct2onnx_expsine.py

    import numpy
    from scipy.spatial.distance import squareform, pdist
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from mlprodict.onnx_grammar import translate_fct2onnx
    from mlprodict.onnx_grammar.onnx_translation import squareform_pdist, py_make_float_array
    from mlprodict.onnxrt import OnnxInference

    # The function to convert into ONNX.
    def kernel_call_ynone(X, length_scale=1.2, periodicity=1.1, pi=3.141592653589793):

        # squareform(pdist(X, ...)) in one function.
        dists = squareform_pdist(X, metric='euclidean')

        # Function starting with 'py_' --> must not be converted into ONNX.
        t_pi = py_make_float_array(pi)
        t_periodicity = py_make_float_array(periodicity)

        # This operator must be converted into ONNX.
        arg = dists / t_periodicity * t_pi
        sin_of_arg = numpy.sin(arg)

        t_2 = py_make_float_array(2)
        t__2 = py_make_float_array(-2)

        t_length_scale = py_make_float_array(length_scale)

        K = numpy.exp((sin_of_arg / t_length_scale) ** t_2 * t__2)
        return K

    # This function is equivalent to the following kernel.
    kernel = ExpSineSquared(length_scale=1.2, periodicity=1.1)

    x = numpy.array([[1, 2], [3, 4]], dtype=float)

    # Checks that the new function and the kernel are the same.
    exp = kernel(x, None)
    got = kernel_call_ynone(x)

    print("ExpSineSquared:")
    print(exp)
    print("numpy function:")
    print(got)

    # Converts the numpy function into an ONNX function.
    fct_onnx = translate_fct2onnx(kernel_call_ynone, cpl=True,
                                  output_names=['Z'])

    # Calls the ONNX function to produce the ONNX algebric function.
    # See below.
    onnx_model = fct_onnx('X')

    # Calls the ONNX algebric function to produce the ONNX graph.
    inputs = {'X': x.astype(numpy.float32)}
    onnx_g = onnx_model.to_onnx(inputs)

    # Creates a python runtime associated to the ONNX function.
    oinf = OnnxInference(onnx_g)

    # Compute the prediction with the python runtime.
    res = oinf.run(inputs)
    print("ONNX output:")
    print(res['Z'])

    # Displays the code of the algebric function.
    print('-------------')
    print("Function code:")
    print('-------------')
    print(translate_fct2onnx(kernel_call_ynone, output_names=['Z']))

The output of function
:func:`translate_fct2onnx
<mlprodict.onnx_grammar.onnx_translation.translate_fct2onnx>`
is not an :epkg:`ONNX` graph but the code of a function which
produces an :epkg:`ONNX` graph. That's why the function is called
twice. The first call compiles the code and a returns a new
:epkg:`python` function. The second call starts all over but
returns the code instead of its compiled version.
