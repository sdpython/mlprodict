"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import timeit
import numpy
import onnx
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.validate_benchmark import make_n_rows


class TestOnnxrtBenchRandomForest(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.6.0')
    @unittest_require_at_least(sklearn, '0.22.0')
    def test_onnxt_iris_random_forest(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(
            X, y, random_state=11, test_size=0.8)
        clr = RandomForestClassifier(n_estimators=10, random_state=42)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test2 = make_n_rows(X_test, 10000)
        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32)

        oinf = OnnxInference(model_def, runtime='python')
        ti = timeit.repeat("oinf.run({'X': X_test2})", number=100,
                           globals={'oinf': oinf, 'X_test2': X_test2},
                           repeat=10)
        self.assertEqual(len(ti), 10)


if __name__ == "__main__":
    unittest.main()
