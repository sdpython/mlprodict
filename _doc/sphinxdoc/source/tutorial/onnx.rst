
.. _l-onnx-tutorial:

Tutorial on ONNX
================

.. contents::
    :local:

Convert a function base on :epkg:`numpy` an :epkg:`scipy`.

.. exref::
    :title: Convert kernel ExpSineSquared from scikit-learn

    This kernel is used by :epkg:`sklearn:gaussian_process:GaussianProcessRegressor`
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

        def kernel_call_ynone(X, length_scale=1.2, periodicity=1.1, pi=3.141592653589793):
            dists = squareform_pdist(X, metric='euclidean')

            t_pi = py_make_float_array(pi)
            t_periodicity = py_make_float_array(periodicity)
            arg = dists / t_periodicity * t_pi

            sin_of_arg = numpy.sin(arg)

            t_2 = py_make_float_array(2)
            t__2 = py_make_float_array(-2)
            t_length_scale = py_make_float_array(length_scale)

            K = numpy.exp((sin_of_arg / t_length_scale) ** t_2 * t__2)
            return K

        kernel = ExpSineSquared(length_scale=1.2, periodicity=1.1)

        x = numpy.array([[1, 2], [3, 4]], dtype=float)

        exp = kernel(x, None)
        got = kernel_call_ynone(x)
        print("ExpSineSquared:")
        print(exp)
        print("numpy function:")
        print(got)

        # converts the numpy function into an ONNX function
        fct_onnx = translate_fct2onnx(kernel_call_ynone, cpl=True,
                                      output_names=['Z'])

        onnx_model = fct_onnx('X')

        # calls the ONNX function to get the ONNX graph
        inputs = {'X': x.astype(numpy.float32)}
        onnx_g = onnx_model.to_onnx(inputs)

        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        print("ONNX output:")
        print(res['Z'])
        print('-------------')
        print("Function code:")
        print('-------------')
        print(translate_fct2onnx(kernel_call_ynone, output_names=['Z']))
