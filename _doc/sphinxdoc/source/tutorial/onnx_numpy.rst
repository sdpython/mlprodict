
.. _l-numpy2onnx-tutorial:

Create custom ONNX graphs with AST
==================================

Converting a :epkg:`scikit-learn` pipeline is easy when
the pipeline contains only pieces implemented in :epkg:`scikit-learn`
associated to a converter in :epkg:`sklearn-onnx`. Outside this
scenario, the conversion usually requires to write custom code
either directly with :epkg:`onnx` operators, either by writing
a `custom converter
<http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/tutorial_2_new_converter.html>`_.
This tutorial addresses a specific scenario involving an instance of
:epkg:`FunctionTransformer`.

.. contents::
    :local:

Translation problem
+++++++++++++++++++

The following pipeline cannot be converted into :epkg:`ONNX` when using
the first examples of `sklearn-onnx tutorial`.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning, FutureWarning

    import numpy
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from skl2onnx import to_onnx

    log_scale_transformer = make_pipeline(
        FunctionTransformer(numpy.log, validate=False),
        StandardScaler())

    X = numpy.random.random((5, 2))

    log_scale_transformer.fit(X)
    print(log_scale_transformer.transform(X))

    # Conversion to ONNX
    try:
        onx = to_onnx(log_scale_transformer, X)
    except (RuntimeError, TypeError) as e:
        print(e)

The first step is a `FunctionTransformer` with a custom function
written with :epkg:`numpy` functions. The pipeline can be converted
only if the function given to this object as argument can be converted
into *ONNX*. Even if function :epkg:`numpy:log` does exist in ONNX specifications
(see `ONNX Log <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log>`_),
this problem is equivalent to a translation from a language, Python,
to another one, ONNX.

Translating numpy to ONNX with AST
++++++++++++++++++++++++++++++++++

.. index:: algebric function

The first approach was to use module :epkg:`ast` to convert
a function into a syntax tree and then try to convert every node
into ONNX to obtain an equivalent ONNX graph.

*mlprodict* implements function
:func:`translate_fct2onnx
<mlprodict.onnx_tools.onnx_grammar.onnx_translation.translate_fct2onnx>`
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
    :warningout: DeprecationWarning, FutureWarning
    :process:
    :store_in_file: fct2onnx_expsine.py

    import numpy
    from scipy.spatial.distance import squareform, pdist
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from mlprodict.onnx_tools.onnx_grammar import translate_fct2onnx
    from mlprodict.onnx_tools.onnx_grammar.onnx_translation import (
        squareform_pdist, py_make_float_array)
    from mlprodict.onnxrt import OnnxInference

    # The function to convert into ONNX.
    def kernel_call_ynone(X, length_scale=1.2, periodicity=1.1,
                          pi=3.141592653589793, op_version=15):

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
    onnx_g = onnx_model.to_onnx(inputs, target_opset=15)

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
<mlprodict.onnx_tools.onnx_grammar.onnx_translation.translate_fct2onnx>`
is not an :epkg:`ONNX` graph but the code of a function which
produces an :epkg:`ONNX` graph. That's why the function is called
twice. The first call compiles the code and a returns a new
:epkg:`python` function. The second call starts all over but
returns the code instead of its compiled version.

This approach has two drawback. The first one is not every function
can be converted into ONNX. That does not mean the algorithm could
not be implemented with ONNX operator. The second drawback is discrepencies.
They should be minimal but still could happen between a numpy and ONNX
implementations.

From ONNX to Python
+++++++++++++++++++

The Python Runtime can be optimized by generating
custom python code and dynamically compile it.
:class:`OnnxInference <mlprodict.onnxrt.onnx_inference.OnnxInference>`
computes predictions based on an ONNX graph with a
python runtime or :epkg:`onnxruntime`.
Method :meth:`to_python
<mlprodict.onnxrt.onnx_inference_exports.OnnxInferenceExport.to_python>`
goes further by converting the ONNX graph into a standalone
python code. All operators may not be implemented.

Another tool is implemented in
`onnx2py.py <https://github.com/microsoft/onnxconverter-common/
blob/master/onnxconverter_common/onnx2py.py>`_ and converts an ONNX
graph into a python code which produces this graph.

Numpy API for ONNX
++++++++++++++++++

This approach fixes the two issues mentioned above. The goal is write
a code using the same function as :epkg:`numpy` offers but
executed by an ONNX runtime. The full API is described at
:ref:`l-numpy-onnxpy` and introduced here.
This section is developped in notebook
:ref:`numpyapionnxrst` and :ref:`l-numpy-api-for-onnx`.
