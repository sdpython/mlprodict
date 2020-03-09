# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_conv import to_onnx
from mlprodict.tools.asv_options_helper import display_onnx


class TestDisplay(ExtTestCase):

    def test_plot_logreg_xtime(self):

        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        lr = LinearRegression()
        lr.fit(X, y)
        model_onnx = to_onnx(lr, X.astype(numpy.float32))
        disp = display_onnx(model_onnx)
        self.assertIn('[...]', disp)
        self.assertIn('opset_import', disp)
        self.assertIn('producer_version', disp)
        self.assertLess(len(disp), 1010)


if __name__ == "__main__":
    unittest.main()
