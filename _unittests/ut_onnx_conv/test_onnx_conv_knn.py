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

    def onnx_test_knn_single_regressor(self, dtype, n_targets=1, debug=False, **kwargs):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = y.astype(dtype)
        if n_targets != 1:
            yn = numpy.empty((y.shape[0], n_targets), dtype=dtype)
            for i in range(n_targets):
                yn[:, i] = y + i
            y = yn
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        X_test = X_test.astype(dtype)
        clr = KNeighborsRegressor(**kwargs)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(dtype),
                            dtype=dtype, rewrite_ops=True)
        oinf = OnnxInference(model_def, runtime='python')

        if debug:
            y = oinf.run({'X': X_test}, verbose=1, fLOG=print)
        else:
            y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        if dtype == numpy.float32:
            self.assertEqualArray(lexp, y['variable'], decimal=5)
        else:
            self.assertEqualArray(lexp, y['variable'])

    def test_onnx_test_knn_single_regressor32(self):
        self.onnx_test_knn_single_regressor(numpy.float32)

    def test_onnx_test_knn_single_regressor32_balltree(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, algorithm='ball_tree')

    def test_onnx_test_knn_single_regressor32_kd_tree(self):
        self.onnx_test_knn_single_regressor(numpy.float32, algorithm='kd_tree')

    def test_onnx_test_knn_single_regressor32_brute(self):
        self.onnx_test_knn_single_regressor(numpy.float32, algorithm='brute')

    def test_onnx_test_knn_single_regressor64(self):
        self.onnx_test_knn_single_regressor(numpy.float64)

    def test_onnx_test_knn_single_regressor32_target2(self):
        self.onnx_test_knn_single_regressor(numpy.float32, n_targets=2)

    def test_onnx_test_knn_single_regressor32_k1(self):
        self.onnx_test_knn_single_regressor(numpy.float32, n_neighbors=1)

    def test_onnx_test_knn_single_regressor32_k1_target2(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, n_neighbors=1, n_targets=2)

    def test_onnx_test_knn_single_regressor32_minkowski(self):
        self.onnx_test_knn_single_regressor(numpy.float32, metric='minkowski')

    @unittest.skip(reason="not yet implemented")
    def test_onnx_test_knn_single_regressor32_distance(self):
        self.onnx_test_knn_single_regressor(numpy.float32, weights='distance')

    @unittest.skip(reason="not yet implemented")
    def test_onnx_test_knn_single_regressor32_minkowski_p3(self):
        self.onnx_test_knn_single_regressor(numpy.float32, metric='minkowski',
                                            metric_params={'p': 3})


if __name__ == "__main__":
    unittest.main()
