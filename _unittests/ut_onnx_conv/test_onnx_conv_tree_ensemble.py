# pylint: disable=R1716
"""
@brief      test log(time=20s)
"""
import unittest
import numpy
from onnx.checker import check_model
from onnxruntime import __version__ as ort_version, InferenceSession
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper.version_helper import compare_module_version
import sklearn
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
import skl2onnx
from mlprodict.onnx_tools.model_check import check_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
# from mlprodict import (
#     __max_supported_opsets_experimental__ as __max_supported_opsets__)
from mlprodict import __max_supported_opsets__

ort_version = ".".join(ort_version.split('.')[:2])


class TestOnnxConvTreeEnsemble(ExtTestCase):

    def common_test_regressor(self, runtime, models=None, dtypes=None):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        if models is None:
            models = [
                DecisionTreeRegressor(max_depth=2),
                RandomForestRegressor(n_estimators=2, max_depth=2),
            ]

            if (compare_module_version(skl2onnx.__version__, "1.11.1") > 0 or
                    compare_module_version(sklearn.__version__, "1.1.0") < 0):
                # "log_loss still not implemented")
                models.append(GradientBoostingRegressor(
                    n_estimators=2, max_depth=2))
                models.append(HistGradientBoostingRegressor(
                    max_iter=2, max_depth=2))

        if dtypes is None:
            dtypes = [numpy.float64, numpy.float32]
        for gbm in models:
            gbm.fit(X_train, y_train)
            exp = gbm.predict(X_test).ravel()
            for dtype in dtypes:
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
                            check_onnx(onx)
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
    def test_regressor_python_lgbm16(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        reg = LGBMRegressor(max_iter=3, max_depth=2, verbosity=-1)
        reg.fit(X_train, y_train)
        try:
            onx = to_onnx(reg, X_train.astype(numpy.float64),
                          target_opset={'': 16, 'ai.onnx.ml': 3},
                          rewrite_ops=True)
        except RuntimeError as e:
            msg = "version 16 of domain '' not supported yet by this library"
            if msg in str(e):
                return
            msg = "version 3 of domain 'ai.onnx.ml' not supported yet"
            if msg in str(e):
                return
            raise e
        node = onx.graph.node[0]
        self.assertEqual(node.op_type, 'TreeEnsembleRegressor')
        self.assertEqual(node.domain, 'ai.onnx.ml')
        set_names = set()
        for att in node.attribute:
            if 'values' in att.name or 'target' in att.name:
                set_names.add(att.name)
        self.assertIn("nodes_values_as_tensor", set_names)
        check_onnx(onx)
        with open("debug.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        # python
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X_test.astype(numpy.float64)})
        self.assertEqual(got['variable'].dtype, numpy.float64)
        # onnxruntime
        sess = InferenceSession(onx.SerializeToString())
        got2 = sess.run(None, {'X': X_test.astype(numpy.float64)})
        self.assertEqual(got2[0].dtype, numpy.float64)

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_python_xgb(self):
        self.common_test_regressor(
            'python', [XGBRegressor(max_iter=3, max_depth=2, verbosity=0)],
            dtypes=[numpy.float32])

    @unittest.skipIf(compare_module_version(ort_version, '1.12') < 0,
                     reason="missing runtime")
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_regressor_onnxruntime(self):
        self.common_test_regressor('onnxruntime1')

    def common_test_classifier(self, runtime, models=None, dtypes=None):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        if models is None:
            models = [
                DecisionTreeClassifier(max_depth=2),
                RandomForestClassifier(n_estimators=2, max_depth=2),
            ]

            if (compare_module_version(skl2onnx.__version__, "1.11.1") > 0 or
                    compare_module_version(sklearn.__version__, "1.1.0") < 0):
                # "log_loss still not implemented")
                models.append(GradientBoostingClassifier(
                    n_estimators=2, max_depth=2))
                models.append(HistGradientBoostingClassifier(
                    max_iter=2, max_depth=2))

        if dtypes is None:
            dtypes = [numpy.float64, numpy.float32]
        for gbm in models:
            gbm.fit(X_train, y_train)
            exp = gbm.predict_proba(X_test).ravel()
            for dtype in dtypes:
                decimal = {numpy.float32: 6, numpy.float64: 7}[dtype]
                if (dtype == numpy.float64 and
                        gbm.__class__ in {DecisionTreeClassifier,
                                          GradientBoostingClassifier}):
                    decimal = 12
                xt = X_test.astype(dtype)
                for opset in [(15, 1), (16, 3)]:
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
                        if dtype == numpy.float64 and (
                                opset[1] >= 3 or
                                gbm.__class__ not in {
                                    RandomForestClassifier,
                                    HistGradientBoostingClassifier}):
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
                            if (dtype != numpy.float64 or
                                    gbm.__class__ == HistGradientBoostingClassifier):
                                # DecisionTree, RandomForest are comparing
                                # a double threshold and a float feature,
                                # the comparison may introduce discrepancies if
                                # the comparison is between both double.
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
        # xgboost is implemented with floats
        self.common_test_classifier(
            'python', [LGBMClassifier(max_iter=3, max_depth=2, verbosity=-1)],
            dtypes=[numpy.float32])

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_classifier_python_xgb(self):
        # xgboost is implemented with floats
        self.common_test_classifier(
            'python', [XGBClassifier(max_iter=2, max_depth=2, verbosity=0)],
            dtypes=[numpy.float32])


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('mlprodict.onnx_conv')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxConvTreeEnsemble().test_regressor_python_lgbm16()
    unittest.main(verbosity=2)
