"""
@brief      test log(time=82s)
"""
import unittest
import numpy
from onnx.checker import check_model
from onnxruntime import __version__ as ort_version
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier)
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict import (
    __max_supported_opsets_experimental__ as __max_supported_opsets__)


ort_version = ".".join(ort_version.split('.')[:2])


class TestOnnxConvTreeEnsemble(ExtTestCase):

    def common_test_regressor(self, runtime, models=None):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        if models is None:
            models = [
                DecisionTreeRegressor(max_depth=2),
                HistGradientBoostingRegressor(max_iter=3, max_depth=2),
                GradientBoostingRegressor(n_estimators=3, max_depth=2),
                RandomForestRegressor(n_estimators=3, max_depth=2),
            ]

        for gbm in models:
            gbm.fit(X_train, y_train)
            exp = gbm.predict(X_test).ravel()
            for dtype in [numpy.float32, numpy.float64]:
                decimal = {numpy.float32: 5, numpy.float64: 12}[dtype]
                if (dtype == numpy.float64 and gbm.__class__ in {
                        LGBMRegressor}):
                    decimal = 7
                elif (dtype == numpy.float64 and gbm.__class__ in {
                        XGBRegressor}):
                    decimal = 7
                xt = X_test.astype(dtype)
                for opset in [(16, 3), (15, 1)]:
                    if opset[1] > __max_supported_opsets__['ai.onnx.ml']:
                        continue
                    with self.subTest(runtime=runtime, dtype=dtype,
                                      model=gbm.__class__.__name__,
                                      opset=opset):
                        onx = to_onnx(gbm, xt,  # options={'zipmap': False},
                                      target_opset={
                                          '': opset[0], 'ai.onnx.ml': opset[1]},
                                      rewrite_ops=True)
                        if dtype == numpy.float64:
                            sonx = str(onx)
                            if 'double' not in sonx and "_as_tensor" not in sonx:
                                raise AssertionError(
                                    "Issue with %s." % str(onx))
                        try:
                            check_model(onx)
                        except Exception as e:
                            raise AssertionError(
                                "Issue with %s." % str(onx)) from e
                        output = onx.graph.output[0].type.tensor_type.elem_type
                        self.assertEqual(
                            output, {numpy.float32: 1, numpy.float64: 11}[dtype])
                        oif = OnnxInference(onx, runtime=runtime)
                        self.assertEqual({numpy.float32: 'tensor(float)',
                                          numpy.float64: 'tensor(double)'}[dtype],
                                         oif.output_names_shapes_types[0][2])
                        got = oif.run({'X': xt})
                        try:
                            self.assertEqualArray(exp, got['variable'].ravel(),
                                                  decimal=decimal)
                        except AssertionError as e:
                            raise AssertionError(
                                "Discrepancies %s." % str(onx)) from e
                        self.assertEqual(got['variable'].dtype, dtype)

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_python(self):
        self.common_test_regressor('python')

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_python_lgbm(self):
        self.common_test_regressor(
            'python', [LGBMRegressor(max_iter=3, max_depth=2, verbosity=-1)])

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_python_xgb(self):
        self.common_test_regressor(
            'python', [XGBRegressor(max_iter=3, max_depth=2, verbosity=0)])

    @unittest.skipIf(compare_module_version(ort_version, '1.12') < 0,
                     reason="missing runtime")
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_onnxruntime(self):
        self.common_test_regressor('onnxruntime1')

    def common_test_classifier(self, runtime, models=None):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        if models is None:
            models = [
                DecisionTreeClassifier(max_depth=2),
                RandomForestClassifier(n_estimators=3, max_depth=2),
                HistGradientBoostingClassifier(max_iter=3, max_depth=2),
                GradientBoostingClassifier(n_estimators=3, max_depth=2),
            ]

        for gbm in models:
            gbm.fit(X_train, y_train)
            exp = gbm.predict_proba(X_test).ravel()
            for dtype in [numpy.float64, numpy.float32]:
                decimal = {numpy.float32: 6, numpy.float64: 7}[dtype]
                if (dtype == numpy.float64 and
                        gbm.__class__ in {DecisionTreeClassifier,
                                          GradientBoostingClassifier}):
                    decimal = 12
                xt = X_test.astype(dtype)
                for opset in [(16, 3), (15, 1)]:
                    if opset[1] > __max_supported_opsets__['ai.onnx.ml']:
                        continue
                    with self.subTest(runtime=runtime, dtype=dtype,
                                      model=gbm.__class__.__name__,
                                      opset=opset):
                        onx = to_onnx(gbm, xt, options={'zipmap': False},
                                      target_opset={
                                          '': opset[0],
                                          'ai.onnx.ml': opset[1]},
                                      rewrite_ops=True)
                        if dtype == numpy.float64:
                            sonx = str(onx)
                            if 'double' not in sonx and "_as_tensor" not in sonx:
                                raise AssertionError(
                                    "Issue with %s." % str(onx))
                        output = onx.graph.output[1].type.tensor_type.elem_type
                        self.assertEqual(
                            output, {numpy.float32: 1, numpy.float64: 11}[dtype])
                        oif = OnnxInference(onx, runtime=runtime)
                        self.assertEqual({numpy.float32: 'tensor(float)',
                                          numpy.float64: 'tensor(double)'}[dtype],
                                         oif.output_names_shapes_types[1][2])
                        got = oif.run({'X': xt})
                        try:
                            self.assertEqualArray(
                                exp, got['probabilities'].ravel(), decimal=decimal)
                        except AssertionError as e:
                            raise AssertionError(
                                "Discrepancies with onx=%s\n%s." % (
                                    onnx_simple_text_plot(onx),
                                    str(onx))) from e
                        self.assertEqual(got['probabilities'].dtype, dtype)

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_classifier_python(self):
        self.common_test_classifier('python')

    @unittest.skipIf(compare_module_version(ort_version, '1.12') < 0,
                     reason="missing runtime")
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_classifier_onnxruntime(self):
        self.common_test_classifier('onnxruntime1')

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_classifier_python_lgbm(self):
        self.common_test_classifier(
            'python', [LGBMClassifier(max_iter=3, max_depth=2, verbosity=-1)])

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_classifier_python_xgb(self):
        self.common_test_classifier(
            'python', [XGBClassifier(max_iter=2, max_depth=2, verbosity=0)])


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('mlprodict.onnx_conv')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxConvTreeEnsemble().test_classifier_python_xgb()
    unittest.main(verbosity=2)
