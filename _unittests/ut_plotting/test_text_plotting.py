# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import textwrap
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsRegressor
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxSub, OnnxDiv, OnnxAbs, OnnxLeakyRelu)
from mlprodict.onnx_conv import to_onnx
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
from mlprodict.plotting.plotting import (
    onnx_text_plot, onnx_text_plot_tree, onnx_simple_text_plot)


class TestPlotTextPlotting(ExtTestCase):

    def test_onnx_text_plot(self):
        idi = numpy.identity(2).astype(numpy.float32)
        opv = get_opset_number_from_onnx()
        A = OnnxAdd('X', idi, op_version=opv)
        B = OnnxSub(A, 'W', output_names=['Y'], op_version=opv)
        onx = B.to_onnx({'X': idi.astype(numpy.float32),
                         'W': idi.astype(numpy.float32)})
        res = onnx_text_plot(onx)
        self.assertIn("Init", res)

    def test_onnx_text_plot_tree(self):
        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        clr = DecisionTreeRegressor(max_depth=3)
        clr.fit(X, y)
        onx = to_onnx(clr, X)
        res = onnx_text_plot_tree(onx.graph.node[0])
        self.assertIn("treeid=0", res)
        self.assertIn("         T y=", res)

    def test_onnx_simple_text_plot_kmeans(self):
        x = numpy.random.randn(10, 3)
        model = KMeans(3)
        model.fit(x)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx)
        expected1 = textwrap.dedent("""
        ReduceSumSquare(X, axes=[1], keepdims=1) -> Re_reduced0
          Mul(Re_reduced0, Mu_Mulcst) -> Mu_C0
            Gemm(X, Ge_Gemmcst, Mu_C0, alpha=-2.00, transB=1) -> Ge_Y0
          Add(Re_reduced0, Ge_Y0) -> Ad_C01
            Add(Ad_Addcst, Ad_C01) -> Ad_C0
              Sqrt(Ad_C0) -> scores
              ArgMin(Ad_C0, axis=1, keepdims=0) -> label
        """).strip(" \n")
        expected2 = textwrap.dedent("""
        ReduceSumSquare(X, axes=[1], keepdims=1) -> Re_reduced0
          Mul(Re_reduced0, Mu_Mulcst) -> Mu_C0
            Gemm(X, Ge_Gemmcst, Mu_C0, alpha=-2.00, transB=1) -> Ge_Y0
          Add(Re_reduced0, Ge_Y0) -> Ad_C01
            Add(Ad_Addcst, Ad_C01) -> Ad_C0
              Sqrt(Ad_C0) -> scores
              ArgMin(Ad_C0, axis=1, keepdims=0) -> label
        """).strip(" \n")
        if expected1 not in text and expected2 not in text:
            raise AssertionError(
                "Unexpected value:\n%s" % text)

    def test_onnx_simple_text_plot_knnr(self):
        x = numpy.random.randn(10, 3)
        y = numpy.random.randn(10)
        model = RadiusNeighborsRegressor(3)
        model.fit(x, y)
        onx = to_onnx(model, x.astype(numpy.float32),
                      target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False)
        expected = "              Neg(arange_y0) -> arange_Y0"
        self.assertIn(expected, text)
        self.assertIn(", to=7)", text)
        self.assertIn(", keepdims=0)", text)
        self.assertIn(", perm=[1,0])", text)

    def test_onnx_simple_text_plot_toy(self):
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        node1 = OnnxAdd('X', x, op_version=15)
        node2 = OnnxSub('X', x, op_version=15)
        node3 = OnnxAbs(node1, op_version=15)
        node4 = OnnxAbs(node2, op_version=15)
        node5 = OnnxDiv(node3, node4, op_version=15)
        node6 = OnnxAbs(node5, output_names=['Y'], op_version=15)
        onx = node6.to_onnx({'X': x.astype(numpy.float32)},
                            outputs={'Y': x}, target_opset=15)
        text = onnx_simple_text_plot(onx, verbose=False)
        expected = textwrap.dedent("""
        Add(X, Ad_Addcst) -> Ad_C0
          Abs(Ad_C0) -> Ab_Y0
        Identity(Ad_Addcst) -> Su_Subcst
          Sub(X, Su_Subcst) -> Su_C0
            Abs(Su_C0) -> Ab_Y02
            Div(Ab_Y0, Ab_Y02) -> Di_C0
              Abs(Di_C0) -> Y
        """).strip(" \n")
        self.assertIn(expected, text)
        text2, out, err = self.capture(
            lambda: onnx_simple_text_plot(onx, verbose=True))
        self.assertEqual(text, text2)
        self.assertIn('BEST:', out)
        self.assertEmpty(err)

    def test_onnx_simple_text_plot_leaky(self):
        x = OnnxLeakyRelu('X', alpha=0.5, op_version=15,
                          output_names=['Y'])
        onx = x.to_onnx({'X': FloatTensorType()},
                        outputs={'Y': FloatTensorType()},
                        target_opset=15)
        text = onnx_simple_text_plot(onx)
        expected = textwrap.dedent("""
        LeakyRelu(X, alpha=0.50) -> Y
        """).strip(" \n")
        self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
