"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum import (
    decompose_einsum_equation, optimize_decompose_einsum_equation)
from mlprodict.onnxrt import OnnxInference


class TestEinsumBug(ExtTestCase):

    def test_abbba(self):
        res = decompose_einsum_equation(
            "ab,b->ba", strategy='numpy', clean=True)
        self.assertNotEmpty(res)

    def test__pprint_forward(self):
        res = decompose_einsum_equation(
            "ab,b->ba", strategy='numpy', clean=True)
        pf = res._pprint_forward()  # pylint: disable=W0212
        spl = pf.split("<- id")
        self.assertEqual(len(spl), 4)

    def common_test_equation(self, equation, dim1, dim2):
        seq = decompose_einsum_equation(
            equation, clean=True, strategy='numpy')
        onx = seq.to_onnx('Y', 'X1', 'X2')
        sequ = equation.replace(",", "_").replace("->", "__")
        with open(f"temp_{sequ}_A.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        a = numpy.random.rand(*list((2, ) * dim1))
        b = numpy.random.rand(*list((2, ) * dim2))
        oinf = OnnxInference(onx)
        got = oinf.run({'X1': a, 'X2': b})
        expected = numpy.einsum(equation, a, b)
        self.assertEqualArray(expected, got['Y'])

        res = optimize_decompose_einsum_equation(
            equation, numpy.float64, optimize=True, runtime="python",
            cache=False, opset=15, decompose=True, strategy='ml',
            verbose=None)
        new_eq = res.equation_
        new_onx = res.onnx_
        sequ = new_eq.replace(",", "_").replace("->", "__")
        with open(f"temp_{sequ}_B.onnx", "wb") as f:
            f.write(new_onx.SerializeToString())
        oinf = OnnxInference(new_onx)
        got = oinf.run({'X0': a, 'X1': b})
        self.assertEqualArray(expected, got['Y'])

    def test_decompose_einsum_abc_cde_abde(self):
        self.common_test_equation("abc,cde->abde", 3, 3)

    def test_decompose_einsum_abcd_cde_abe(self):
        self.common_test_equation("abcd,cde->abe", 4, 3)


if __name__ == "__main__":
    unittest.main()
