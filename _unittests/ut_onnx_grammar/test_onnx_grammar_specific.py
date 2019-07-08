"""
@brief      test log(time=4s)
"""
import unittest
import numpy
from scipy.spatial.distance import squareform, pdist
from pyquickhelper.pycode import ExtTestCase
from sklearn.gaussian_process.kernels import ExpSineSquared
from mlprodict.onnx_grammar import translate_fct2onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxGrammarSpecific(ExtTestCase):

    def test_export_sklearn_kernel_error(self):

        length_scale = 3
        periodicity = 4
        numpy_pi = numpy.pi

        def squareform_pdist(X, metric='sqeuclidean'):
            return squareform(pdist(X, metric=metric))

        def kernel_call_ynone(X, length_scale=length_scale, periodicity=periodicity):
            dists = squareform_pdist(X, metric='euclidean')
            arg = numpy_pi * dists / periodicity
            sin_of_arg = numpy.sin(arg)
            K = numpy.exp(- 2 * (sin_of_arg / length_scale) ** 2)
            return K

        context = {'numpy.sin': numpy.sin, 'numpy.exp': numpy.exp,
                   'numpy_pi': numpy.pi, 'squareform_pdist': 'squareform_pdist'}

        self.assertRaise(lambda: translate_fct2onnx(kernel_call_ynone, context=context,
                                                    output_names=['Z']), RuntimeError)

    def test_export_sklearn_kernel_neg(self):

        x = numpy.array([[1, 2], [3, 4]], dtype=float)

        kernel = ExpSineSquared(length_scale=1.2, periodicity=1.1)

        def squareform_pdist(X, metric='sqeuclidean'):
            return squareform(pdist(X, metric=metric))

        def kernel_call_ynone(X, length_scale=1.2, periodicity=1.1, pi=3.141592653589793):
            dists = squareform_pdist(X, metric='euclidean')
            arg = dists / periodicity * pi
            sin_of_arg = numpy.sin(arg)
            K = numpy.exp((sin_of_arg / length_scale) ** 2 * (-2))
            return K

        exp = kernel(x, None)
        got = kernel_call_ynone(x)
        self.assertEqualArray(exp, got)
        context = {'numpy.sin': numpy.sin, 'numpy.exp': numpy.exp,
                   'numpy_pi': numpy.pi, 'squareform_pdist': 'squareform_pdist'}

        onnx_code = translate_fct2onnx(kernel_call_ynone, context=context,
                                       output_names=['Z'])
        self.assertIn(
            "X, length_scale=1.2, periodicity=1.1, pi=3.14159", onnx_code)
        self.assertIn("-2", onnx_code)
        self.assertIn('metric="euclidean"', onnx_code)

    def test_export_sklearn_kernel(self):

        x = numpy.array([[1, 2], [3, 4]], dtype=float)

        kernel = ExpSineSquared(length_scale=1.2, periodicity=1.1)

        def squareform_pdist(X, metric='sqeuclidean'):
            return squareform(pdist(X, metric=metric))

        def py_make_float_array(cst):
            return numpy.array([cst], dtype=numpy.float32)

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

        exp = kernel(x, None)
        got = kernel_call_ynone(x)
        self.assertEqualArray(exp, got)
        context = {'numpy.sin': numpy.sin, 'numpy.exp': numpy.exp,
                   'numpy_pi': numpy.pi, 'squareform_pdist': 'squareform_pdist',
                   'py_make_float_array': py_make_float_array}

        onnx_code = translate_fct2onnx(kernel_call_ynone, context=context,
                                       output_names=['Z'])
        self.assertIn(
            "X, length_scale=1.2, periodicity=1.1, pi=3.14159", onnx_code)
        self.assertIn("-2", onnx_code)
        self.assertIn('metric="euclidean"', onnx_code)

        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxSin, OnnxMul, OnnxIdentity, OnnxPow, OnnxDiv, OnnxExp
        )
        from skl2onnx.algebra.complex_functions import squareform_pdist as Onnxsquareform_pdist
        ctx = {'OnnxAdd': OnnxAdd, 'OnnxPow': OnnxPow,
               'OnnxSin': OnnxSin, 'OnnxDiv': OnnxDiv,
               'OnnxMul': OnnxMul, 'OnnxIdentity': OnnxIdentity,
               'OnnxExp': OnnxExp,
               'Onnxsquareform_pdist': Onnxsquareform_pdist,
               'py_make_float_array': py_make_float_array}

        fct = translate_fct2onnx(kernel_call_ynone, context=context,
                                 cpl=True, context_cpl=ctx,
                                 output_names=['Z'])

        r = fct('X')
        self.assertIsInstance(r, OnnxIdentity)

        inputs = {'X': x.astype(numpy.float32)}
        onnx_g = r.to_onnx(inputs)
        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])


if __name__ == "__main__":
    unittest.main()
