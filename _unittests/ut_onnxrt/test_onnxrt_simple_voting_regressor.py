"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtSimpleVotingRegressor(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_iris_voting_regressor(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        clr = VotingRegressor(
            estimators=[
                ('lr', LinearRegression()),
                ('dt', DecisionTreeRegressor(max_depth=2))
            ])
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:4], X_test[-4:]])
        res0 = clr.predict(X_test).astype(numpy.float32)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))

        oinf = OnnxInference(model_def, runtime='python')
        res1 = oinf.run({'X': X_test})
        regs = DataFrame(res1['variable']).values
        self.assertEqualArray(res0, regs.ravel(), decimal=6)


if __name__ == "__main__":
    unittest.main()
