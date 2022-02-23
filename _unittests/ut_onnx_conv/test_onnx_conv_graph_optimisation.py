"""
@brief      test log(time=3s)
"""
from collections import OrderedDict
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.onnx_conv.scorers.cdist_score import score_cdist_sum


class TestOnnxConvGraphOptimisation(ExtTestCase):

    def test_to_onnx_rename_names(self):
        data = load_iris()
        X, y = data.data, data.target
        model = KNeighborsRegressor(n_neighbors=2).fit(X, y)

        model_onnx = to_onnx(
            model, X[:1], target_opset=TARGET_OPSET)
        oinf1 = OnnxInference(model_onnx)
        y1 = oinf1.run({'X': X})['variable']

        model_onnx = to_onnx(
            model, X[:1], target_opset=TARGET_OPSET,
            rename_strategy='simple')
        oinf1 = OnnxInference(model_onnx)
        y2 = oinf1.run({'X': X})['variable']
        self.assertEqualArray(y1, y2)

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_to_onnx_rename_names_scorer(self):
        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        Y = X[:2].copy()
        Y[0, :] = 0

        init_types = OrderedDict([('X', X), ('Y', Y)])
        opset = TARGET_OPSET
        scorer = make_scorer(
            score_cdist_sum, metric='sqeuclidean',
            greater_is_better=False)

        monx1 = to_onnx(scorer, init_types, target_opset=opset,
                        rewrite_ops=True)
        monx2 = to_onnx(scorer, init_types, target_opset=opset,
                        rewrite_ops=True, rename_strategy='simple')

        oinf1 = OnnxInference(monx1)
        oinf2 = OnnxInference(monx2)
        res0 = score_cdist_sum(X, Y, metric='sqeuclidean')
        res1 = oinf1.run({'X': X, 'Y': Y})['scores']
        res2 = oinf2.run({'X': X, 'Y': Y})['scores']
        self.assertEqualArray(res1, res0, decimal=5)
        self.assertEqualArray(res2, res0, decimal=5)


if __name__ == "__main__":
    unittest.main()
