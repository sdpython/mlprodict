"""
@brief      test log(time=400s)
"""
import unittest
import numpy
from onnxruntime import __version__ as ort_version
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, ignore_warnings
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx


ort_version = ".".join(ort_version.split('.')[:2])


class TestOnnxConvTreeEnsemble(ExtTestCase):

    def common_test_regressor(self, runtime):
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
                for opset in [1, 3]:
                    onx = to_onnx(gbm, xt, # options={'zipmap': False},
                                  target_opset={'': 16, 'ai.onnx.ml': opset},
                                  rewrite_ops=True)
                    with self.subTest(runtime=runtime, dtype=dtype,
                                      model=gbm.__class__.__name__,
                                      opset=opset):
                        oif = OnnxInference(onx, runtime=runtime)
                        got = oif.run({'X': X_test})
                        self.assertEqualArray(exp, got['variable'].ravel())
                        self.assertEqual(got['variable'].dtype, dtype)

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_python(self):
        self.common_test_regressor('python')

    @unittest.skipIf(compare_module_version(ort_version, '1.12') < 0,
                     reason="missing runtime")
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_onnxruntime(self):
        self.common_test_regressor('onnxruntime1')
        

if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('mlprodict.onnx_conv')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxConvTreeEnsemble().test_regressor()
    unittest.main()
