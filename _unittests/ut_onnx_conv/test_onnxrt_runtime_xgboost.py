"""
@brief      test log(time=5s)
"""
import sys
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, ignore_warnings
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict.plotting.plotting import onnx_text_plot_tree
from mlprodict import __max_supported_opsets__


TARGET_OPSET = __max_supported_opsets__


def fct_cl2(y):
    y[y == 2] = 0
    return y


def fct_cl3(y):
    y[y == 0] = 6
    return y


def fct_id(y):
    return y


class TestOnnxrtRuntimeXGBoost(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', reason='stuck')
    @ignore_warnings(UserWarning)
    def test_onnxrt_python_xgbregressor(self):
        from xgboost import XGBRegressor, XGBClassifier  # pylint: disable=C0411
        obj_classes = {
            'reg:logistic': (XGBClassifier, fct_cl2,
                             make_classification(n_features=4, n_classes=2,
                                                 n_clusters_per_class=1)),
            'binary:logistic': (XGBClassifier, fct_cl2,
                                make_classification(n_features=4, n_classes=2,
                                                    n_clusters_per_class=1)),
            'multi:softmax': (XGBClassifier, fct_id,
                              make_classification(n_features=4, n_classes=3,
                                                  n_clusters_per_class=1)),
            'multi:softprob': (XGBClassifier, fct_id,
                               make_classification(n_features=4, n_classes=3,
                                                   n_clusters_per_class=1)),
            'reg:squarederror': (XGBRegressor, fct_id,
                                 make_regression(n_features=4, n_targets=1)),
            'reg:squarederror2': (XGBRegressor, fct_id,
                                  make_regression(n_features=4, n_targets=2)),
        }
        nb_tests = 0
        for objective in obj_classes:  # pylint: disable=C0206
            for n_estimators in [1, 2]:
                with self.subTest(objective=objective, n_estimators=n_estimators):
                    probs = []
                    cl, fct, prob = obj_classes[objective]

                    iris = load_iris()
                    X, y = iris.data, iris.target
                    y = fct(y)
                    X_train, X_test, y_train, _ = train_test_split(
                        X, y, random_state=11)
                    probs.append((X_train, X_test, y_train))

                    X_train, X_test, y_train, _ = train_test_split(
                        *prob, random_state=11)
                    probs.append((X_train, X_test, y_train))

                    for X_train, X_test, y_train in probs:
                        obj = objective.replace(
                            'reg:squarederror2', 'reg:squarederror')
                        obj = obj.replace(
                            'multi:softmax2', 'multi:softmax')
                        clr = cl(objective=obj, n_estimators=n_estimators)
                        if len(y_train.shape) == 2:
                            y_train = y_train[:, 1]
                        try:
                            clr.fit(X_train, y_train)
                        except ValueError as e:
                            raise AssertionError(
                                "Unable to train with objective %r and data %r." % (
                                    objective, y_train)) from e

                        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                                            target_opset=TARGET_OPSET)

                        oinf = OnnxInference(model_def)
                        y = oinf.run({'X': X_test.astype(numpy.float32)})
                        if cl == XGBRegressor:
                            exp = clr.predict(X_test)
                            self.assertEqual(list(sorted(y)), ['variable'])
                            self.assertEqualArray(
                                exp, y['variable'].ravel(), decimal=5)
                        else:
                            if 'softmax' not in obj:
                                exp = clr.predict_proba(X_test)
                                self.assertEqual(list(sorted(y)), [
                                                 'output_label', 'output_probability'])
                                got = DataFrame(y['output_probability']).values
                                self.assertEqualArray(exp, got, decimal=5)

                            exp = clr.predict(X_test[:10])
                            self.assertEqualArray(exp, y['output_label'][:10])

                        nb_tests += 1

        self.assertGreater(nb_tests, 8)

    @ignore_warnings(UserWarning)
    @unittest.skipIf(sys.platform == 'darwin', reason='stuck')
    def test_xgboost_classifier_i5450(self):
        from xgboost import XGBClassifier  # pylint: disable=C0411
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=10)
        clr = XGBClassifier(objective="multi:softprob",
                            max_depth=1, n_estimators=2)
        clr.fit(X_train, y_train, eval_set=[
                (X_test, y_test)], early_stopping_rounds=40)
        onx = to_onnx(clr, X_train[:1].astype(numpy.float32),
                      options={XGBClassifier: {'zipmap': False}},
                      target_opset=TARGET_OPSET)
        sess = OnnxInference(onx)
        predict_list = [1., 20., 466., 0.]
        predict_array = numpy.array(predict_list).reshape(
            (1, -1)).astype(numpy.float32)
        pred_onx = sess.run({'X': predict_array})
        pred_onx = pred_onx['probabilities']
        pred_xgboost = clr.predict_proba(predict_array)
        self.assertEqualArray(pred_xgboost, pred_onx)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', reason='stuck')
    @ignore_warnings(UserWarning)
    def test_onnxrt_python_xgbclassifier(self):
        from xgboost import XGBClassifier  # pylint: disable=C0411
        x = numpy.random.randn(100, 10).astype(numpy.float32)
        y = ((x.sum(axis=1) + numpy.random.randn(x.shape[0]) / 50 + 0.5) >= 0).astype(numpy.int64)
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        bmy = numpy.mean(y_train)
        
        for bm in [None, bmy]:
            with self.subTest(base_score=bm):
                model_skl = XGBClassifier(n_estimators=1, 
                                          learning_rate=0.01,
                                          subsample=0.5, objective="binary:logistic",
                                          base_score=bm, max_depth=2)
                model_skl.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)

                model_onnx_skl = to_onnx(model_skl, x_train, rewrite_ops=True,
                                        target_opset={'': 15, 'ai.onnx.ml': 2},
                                        options={'zipmap': False})    

                oinf = OnnxInference(model_onnx_skl)
                res2 = oinf.run({'X': x_test})
                dump = model_skl.get_booster().get_dump()

                print(bm)
                from pprint import pprint
                pprint(model_skl.get_xgb_params())
                print("\n".join(dump))
                print(onnx_text_plot_tree(model_onnx_skl.graph.node[0]))
                self.assertEqualArray(model_skl.predict_proba(x_test),
                                      res2['probabilities'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
