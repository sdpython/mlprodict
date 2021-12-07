"""
@brief      test log(time=5s)
"""
import unittest
import textwrap
import numpy
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsRegressor
from skl2onnx.algebra.onnx_ops import (
    OnnxAdd, OnnxSub, OnnxDiv, OnnxAbs)
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx_tools import simple_onnx_str
from mlprodict.onnx_conv import to_onnx


class TestOnnxTools(ExtTestCase):

    def test_simple_onnx_str_kmeans(self):
        x = numpy.random.randn(10, 3)
        model = KMeans(3)
        model.fit(x)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = simple_onnx_str(onx)
        expected = textwrap.dedent("""
        ReduceSumSquare(X) -> Re_reduced0
          Mul(Re_reduced0, Mu_Mulcst) -> Mu_C0
            Gemm(X, Ge_Gemmcst, Mu_C0) -> Ge_Y0
        ---
          Add(Re_reduced0, Ge_Y0) -> Ad_C01
            Add(Ad_Addcst, Ad_C01) -> Ad_C0
              Sqrt(Ad_C0) -> scores
              ArgMin(Ad_C0) -> label
        """).strip(" \n")
        self.assertIn(expected, text)

    def test_simple_onnx_str_knnr(self):
        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = RadiusNeighborsRegressor(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = simple_onnx_str(onx)
        self.assertIn(
            "output: name='variable' type=dtype('float32') shape=(0, 1)",
            text)

    def test_simple_onnx_str_toy(self):
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        node1 = OnnxAdd('X', x, op_version=15)
        node2 = OnnxSub('X', x, op_version=15)
        node3 = OnnxAbs(node1, op_version=15)
        node4 = OnnxAbs(node2, op_version=15)
        node5 = OnnxDiv(node3, node4, op_version=15)
        node6 = OnnxAbs(node5, output_names=['Y'], op_version=15)
        onx = node6.to_onnx({'X': x.astype(numpy.float32)},
                            outputs={'Y': x}, target_opset=15)
        text = simple_onnx_str(onx)
        self.assertIn(
            "output: name='variable' type=dtype('float32') shape=(0, 1)",
            text)


if __name__ == "__main__":
    unittest.main()
