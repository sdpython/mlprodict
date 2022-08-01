# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import os
import warnings
import unittest
import numpy
from pyquickhelper.pycode import (
    ExtTestCase, skipif_travis, skipif_circleci, get_temp_folder,
    skipif_appveyor)
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skl2onnx.algebra.onnx_ops import OnnxConcat  # pylint: disable=E0611
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.plotting import plot_onnx


class TestPlotOnnx(ExtTestCase):

    @skipif_travis('graphviz is not installed')
    @skipif_circleci('graphviz is not installed')
    @skipif_appveyor('graphviz is not installed')
    def test_plot_onnx(self):

        cst = numpy.array([[1, 2]], dtype=numpy.float32)
        onx = OnnxConcat('X', 'Y', cst, output_names=['Z'],
                         op_version=12)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        Y = numpy.array([[8, 9], [10, 11], [12, 13]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32),
                                 'Y': Y.astype(numpy.float32)},
                                outputs=[('Z', FloatTensorType([2]))],
                                target_opset=12)

        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1)

        try:
            plot_onnx(model_def, ax=ax)
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                warnings.warn(
                    "Unable to test the dot syntax, dot is mssing", UserWarning)
                return
            raise e
        if __name__ == "__main__":
            temp = get_temp_folder(__file__, "temp_plot_onnx")
            img = os.path.join(temp, "img.png")
            plt.savefig(img)
            plt.show()
        plt.close('all')

    @skipif_travis('graphviz is not installed')
    @skipif_circleci('graphviz is not installed')
    @skipif_appveyor('graphviz is not installed')
    def test_plot_onnx_function(self):
        data = load_iris()
        X, y = data.data, data.target
        steps = [
            ("preprocessing", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty='l1', solver="liblinear"))]
        pipe = Pipeline(steps)
        pipe.fit(X, y)
        onxf = to_onnx(pipe, X, as_function=True, options={'zipmap': False})

        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1)

        try:
            plot_onnx(onxf, ax=ax)
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                warnings.warn(
                    "Unable to test the dot syntax, dot is mssing", UserWarning)
                return
            raise e
        if __name__ == "__main__":
            temp = get_temp_folder(__file__, "temp_plot_onnx_functions")
            img = os.path.join(temp, "img.png")
            plt.savefig(img)
            plt.show()
        plt.close('all')


if __name__ == "__main__":
    # TestPlotOnnx().test_plot_onnx_function()
    unittest.main()
