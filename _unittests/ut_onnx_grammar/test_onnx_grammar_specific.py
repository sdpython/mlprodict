"""
@brief      test log(time=4s)
"""
import unittest
import numpy
from scipy.spatial.distance import squareform, pdist
from pyquickhelper.pycode import ExtTestCase
from sklearn.gaussian_process.kernels import ExpSineSquared
from mlprodict.onnx_grammar import translate_fct2onnx
# from mlprodict.onnxrt import OnnxInference


class TestOnnxGrammarSpecific(ExtTestCase):

    def test_export_sklearn_kernel(self):

        x = numpy.array([[1, 2], [3, 4]], dtype=float)

        kernel = ExpSineSquared()
        length_scale = kernel.length_scale
        periodicity = kernel.periodicity
        numpy_pi = numpy.pi

        def squareform_pdist(X, metric='sqeuclidean'):
            return squareform(pdist(X, metric=metric))

        def kernel_call_ynone(X, length_scale=length_scale, periodicity=periodicity):
            dists = squareform_pdist(X, metric='euclidean')
            arg = numpy_pi * dists / periodicity
            sin_of_arg = numpy.sin(arg)
            K = numpy.exp(- 2 * (sin_of_arg / length_scale) ** 2)
            return K

        exp = kernel(x, None)
        got = kernel_call_ynone(x)
        self.assertEqualArray(exp, got)
        context = {'numpy.sin': numpy.sin, 'numpy.exp': numpy.exp,
                   'numpy_pi': numpy.pi, 'squareform_pdist': 'squareform_pdist'}

        onnx_code = translate_fct2onnx(kernel_call_ynone, context=context,
                                       output_names=['Z'])

        self.assertIn('metric="euclidean"', onnx_code)
        """
        print(onnx_code)

        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxAdd, OnnxTranspose, OnnxMul, OnnxIdentity
        )
        ctx = {'OnnxAdd': OnnxAdd,
               'OnnxTranspose': OnnxTranspose,
               'OnnxMul': OnnxMul, 'OnnxIdentity': OnnxIdentity}

        fct = translate_fct2onnx(kernel_call_ynone, context=context,
                                 cpl=True, context_cpl=ctx,
                                 output_names=['Z'])

        r = fct('x')
        self.assertIsInstance(r, OnnxIdentity)

        inputs = {'x': x.astype(numpy.float32)}
        onnx_g = r.to_onnx(inputs)
        oinf = OnnxInference(onnx_g)
        res = oinf.run(inputs)
        self.assertEqualArray(exp, res['Z'])
        """  # pylint: disable=W0105


if __name__ == "__main__":
    unittest.main()
