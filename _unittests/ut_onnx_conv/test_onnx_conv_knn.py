"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import warnings
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from mlprodict.onnx_conv import register_converters
from mlprodict.onnxrt import OnnxInference, to_onnx


class TestOnnxConvKNN(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_register_converters(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_converters(True)
        self.assertGreater(len(res), 2)

    def onnx_test_knn_single_regressor(self, dtype):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = y.astype(dtype)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        X_test = X_test.astype(dtype)
        clr = KNeighborsRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(dtype),
                            dtype=dtype, rewrite_ops=True)
        oinf = OnnxInference(model_def, runtime='python')

        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        if dtype == numpy.float32:
            self.assertEqualArray(lexp, y['variable'].ravel(), decimal=5)
        else:
            self.assertEqualArray(lexp, y['variable'].ravel())

    def test_onnx_test_knn_single_regressor32(self):
        self.onnx_test_knn_single_regressor(numpy.float32)


if __name__ == "__main__":
    unittest.main()
