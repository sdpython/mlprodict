# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxSub  # pylint: disable=E0611
from mlprodict.onnx_conv import to_onnx
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx
from mlprodict.plotting.plotting import onnx_text_plot, onnx_text_plot_tree


class TestPlotTextPlotting(ExtTestCase):

    def test_onnx_text_plot(self):
        idi = numpy.identity(2)
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


if __name__ == "__main__":
    unittest.main()
