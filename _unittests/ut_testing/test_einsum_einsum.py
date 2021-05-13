"""
@brief      test log(time=8s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum import einsum
from mlprodict.testing.einsum.einsum_fct import enumerate_cached_einsum
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx


class TestEinsumEinsum(ExtTestCase):

    def common_test(self, equation, runtime=None, opset=None, N=5,
                    optimize=False, decompose=True):
        if opset is None:
            opset = get_opset_number_from_onnx()
        inps = equation.split('->')[0].split(',')
        lens = [len(s) for s in inps]
        inputs = [numpy.random.randn(N ** d).reshape((N,) * d)
                  for d in lens]
        if runtime is None:
            if decompose:
                runtime = ['batch_dot', 'python', 'onnxruntime1']
            else:
                runtime = ['python', 'onnxruntime1']
        elif isinstance(runtime, str):
            runtime = [runtime]
        for rt in runtime:
            for dtype in [numpy.float32, numpy.float64]:
                decimal = 5 if dtype == numpy.float32 else 8
                with self.subTest(dt=dtype, rt=rt,
                                  eq=equation, opset=opset,
                                  opt=optimize,
                                  decompose=decompose):
                    typed = [i.astype(dtype) for i in inputs]
                    kwargs = dict(runtime=rt, opset=opset, optimize=optimize,
                                  decompose=decompose)
                    if __name__ == "__main__":
                        kwargs["verbose"] = 1
                    exp = numpy.einsum(equation, *typed)
                    got = einsum(equation, *typed, **kwargs)
                    self.assertEqualArray(exp, got, decimal=decimal)
                    got = einsum(equation, *typed, **kwargs)
                    self.assertEqualArray(exp, got, decimal=decimal)

    def test_einsum(self):
        self.common_test("abc,cd->abd")
        self.common_test("abc,cd,de->abe")
        res = list(enumerate_cached_einsum())
        self.assertGreater(len(res), 2)
        self.assertIn('CachedEinsum', str(res))

    def test_einsum_optimize(self):
        self.common_test("abc,cd->abd", optimize=True)

    def test_einsum_optimize_no(self):
        self.common_test("abc,cd->abd", optimize=True, decompose=False)


if __name__ == "__main__":
    unittest.main()
