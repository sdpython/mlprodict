"""
@brief      test log(time=4s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.gaussian_process.kernels import ExpSineSquared, DotProduct, RationalQuadratic
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx.algebra.onnx_ops import OnnxIdentity  # pylint: disable=E0611
from mlprodict.onnx_grammar import translate_fct2onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_grammar.onnx_translation import get_default_context, get_default_context_cpl
from mlprodict.onnx_grammar.onnx_translation import (
    py_make_float_array, py_pow, squareform_pdist, py_mul, py_opp
)


threshold = "1.5.0"


class TestOnnxGrammarSpecific(ExtTestCase):

    def test_export_sklearn_kernel_error(self):

        length_scale = 3
        periodicity = 4
        numpy_pi = numpy.pi

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

    def test_export_sklearn_kernel_error_prefix(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist

        def kernel_call_ynone(X, length_scale=1.2, periodicity=1.1, pi=3.141592653589793):
            dists = onnx_squareform_pdist(X, metric='euclidean')
            arg = dists / periodicity * pi
            sin_of_arg = numpy.sin(arg)
            K = numpy.exp((sin_of_arg / length_scale) ** 2 * (-2))
            return K

        self.assertRaise(lambda: translate_fct2onnx(kernel_call_ynone, output_names=['Z']),
                         RuntimeError, "'onnx_'")

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="missing complex functions")
    def test_export_sklearn_kernel_exp_sine_squared(self):

        x = numpy.array([[1, 2], [3, 4]], dtype=float)

        kernel = ExpSineSquared(length_scale=1.2, periodicity=1.1)

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

        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611,E0401
            OnnxAdd, OnnxSin, OnnxMul, OnnxPow, OnnxDiv, OnnxExp
        )
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        ctx = {'OnnxAdd': OnnxAdd, 'OnnxPow': OnnxPow,
               'OnnxSin': OnnxSin, 'OnnxDiv': OnnxDiv,
               'OnnxMul': OnnxMul, 'OnnxIdentity': OnnxIdentity,
               'OnnxExp': OnnxExp,
               'onnx_squareform_pdist': onnx_squareform_pdist,
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

    def test_export_sklearn_kernel_dot_product(self):

        def kernel_call_ynone(X, sigma_0=2.):
            t_sigma_0 = py_make_float_array(py_pow(sigma_0, 2))
            K = X @ numpy.transpose(X, axes=[1, 0]) + t_sigma_0
            return K

        x = numpy.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        kernel = DotProduct(sigma_0=2.)
        exp = kernel(x, None)
        got = kernel_call_ynone(x, sigma_0=2.)
        self.assertEqualArray(exp, got)

        context = {'numpy.inner': numpy.inner, 'numpy.transpose': numpy.transpose,
                   'py_pow': py_pow,
                   'py_make_float_array': py_make_float_array}

        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611,E0401
            OnnxTranspose, OnnxMatMul, OnnxAdd, OnnxPow
        )
        ctx = {'OnnxPow': OnnxPow, 'OnnxAdd': OnnxAdd,
               'OnnxIdentity': OnnxIdentity, 'OnnxTranspose': OnnxTranspose,
               'OnnxMatMul': OnnxMatMul,
               'py_make_float_array': py_make_float_array,
               'py_pow': py_pow}

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

        exp = kernel(x.T, None)
        got = kernel_call_ynone(x.T)
        self.assertEqualArray(exp, got)
        inputs = {'X': x.T.astype(numpy.float32)}
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])

    def test_default_context(self):
        ctx = get_default_context()
        self.assertGreater(len(ctx), 10)

    def test_default_context_cpl(self):
        ctx = get_default_context_cpl()
        self.assertGreater(len(ctx), 10)
        self.assertIn('OnnxAdd', ctx)
        self.assertIn('OnnxAdd', str(ctx['OnnxAdd']))

    def test_export_sklearn_kernel_dot_product_default(self):

        def kernel_call_ynone(X, sigma_0=2.):
            t_sigma_0 = py_make_float_array(py_pow(sigma_0, 2))
            K = X @ numpy.transpose(X, axes=[1, 0]) + t_sigma_0
            return K

        x = numpy.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        kernel = DotProduct(sigma_0=2.)
        exp = kernel(x, None)
        got = kernel_call_ynone(x, sigma_0=2.)
        self.assertEqualArray(exp, got)

        fct = translate_fct2onnx(
            kernel_call_ynone, cpl=True, output_names=['Z'])

        r = fct('X')
        self.assertIsInstance(r, OnnxIdentity)
        inputs = {'X': x.astype(numpy.float32)}
        onnx_g = r.to_onnx(inputs)
        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])

        exp = kernel(x.T, None)
        got = kernel_call_ynone(x.T)
        self.assertEqualArray(exp, got)
        inputs = {'X': x.T.astype(numpy.float32)}
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])

    @unittest.skipIf(compare_module_version(skl2onnx_version, threshold) <= 0,
                     reason="missing complex functions")
    def test_export_sklearn_kernel_rational_quadratic(self):

        def kernel_rational_quadratic_none(X, length_scale=1.0, alpha=2.0):
            dists = squareform_pdist(X, metric='sqeuclidean')
            cst = py_pow(length_scale, 2)
            cst = py_mul(cst, alpha, 2)
            t_cst = py_make_float_array(cst)
            tmp = dists / t_cst
            t_one = py_make_float_array(1)
            base = tmp + t_one
            t_alpha = py_make_float_array(py_opp(alpha))
            K = numpy.power(base, t_alpha)
            return K

        x = numpy.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        kernel = RationalQuadratic(length_scale=1.0, alpha=2.0)
        exp = kernel(x, None)
        got = kernel_rational_quadratic_none(x, length_scale=1.0, alpha=2.0)
        self.assertEqualArray(exp, got)

        fct = translate_fct2onnx(
            kernel_rational_quadratic_none, cpl=True, output_names=['Z'])

        r = fct('X')
        self.assertIsInstance(r, OnnxIdentity)
        inputs = {'X': x.astype(numpy.float32)}
        onnx_g = r.to_onnx(inputs)
        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])

        exp = kernel(x.T, None)
        got = kernel_rational_quadratic_none(x.T)
        self.assertEqualArray(exp, got)
        inputs = {'X': x.T.astype(numpy.float32)}
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])


if __name__ == "__main__":
    unittest.main()
