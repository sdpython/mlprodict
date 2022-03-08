"""
@brief      test log(time=400s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx


class TestOnnxConvTreeEnsemble(ExtTestCase):

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        models = [
            RandomForestRegressor(n_estimators=3, max_depth=2)]
        
        for gbm in models:
            gbm.fit(X_train, y_train)
            exp = gbm.predict(X_test).ravel()
            for dtype in [numpy.float64, numpy.float32]:
                xt = X.astype(dtype)
                onx = to_onnx(gbm, xt, # options={'zipmap': False},
                              target_opset={'': 16, 'ai.onnx.ml': 3},
                              rewrite_ops=True)
                for rt in ['python', 'onnxruntime1']:
                    with self.subTest(runtime=rt, dtype=dtype,
                                      model=gbm.__class__.__name__):
                        oif = OnnxInference(onx, runtime=rt)
                        got = oif.run({'X': X})
                        self.assertEqualArray(exp, got['probabilities'].ravel())
                        self.assertEqual(got['probabilities'].dtype, dtype)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('mlprodict.onnx_conv')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    TestOnnxConvTreeEnsemble().test_regressor()
    unittest.main()
