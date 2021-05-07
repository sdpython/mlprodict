"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from xgboost import XGBRegressor, XGBClassifier  # pylint: disable=C0411
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx


def fct_cl2(y):
    y[y == 2] = 0
    return y


def fct_cl3(y):
    y[y == 0] = 6
    return y


def fct_id(y):
    return y


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
    'multi:softmax2': (XGBClassifier, fct_cl3,
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


class TestOnnxrtRuntimeXGBoost(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @skipif_circleci('stuck')
    def test_onnxrt_python_xgbregressor(self):
        nb_tests = 0
        for objective in obj_classes:
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
                        clr = cl(objective=obj, n_estimators=n_estimators)
                        if len(y_train.shape) == 2:
                            y_train = y_train[:, 1]
                        clr.fit(X_train, y_train)

                        model_def = to_onnx(clr, X_train.astype(numpy.float32))

                        oinf = OnnxInference(model_def)
                        y = oinf.run({'X': X_test.astype(numpy.float32)})
                        if cl == XGBRegressor:
                            exp = clr.predict(X_test)
                            self.assertEqual(list(sorted(y)), ['variable'])
                            self.assertEqualArray(
                                exp, y['variable'].ravel(), decimal=5)
                        else:
                            exp = clr.predict_proba(X_test)
                            self.assertEqual(list(sorted(y)), [
                                             'output_label', 'output_probability'])
                            got = DataFrame(y['output_probability']).values
                            self.assertEqualArray(exp, got, decimal=5)

                            exp = clr.predict(X_test)
                            self.assertEqualArray(exp, y['output_label'])

                        nb_tests += 1

        self.assertGreater(nb_tests, 8)

    def test_xgboost_classifier_i5450(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=10)
        clr = XGBClassifier(objective="multi:softmax",
                            max_depth=1, n_estimators=2)
        clr.fit(X_train, y_train, eval_set=[
                (X_test, y_test)], early_stopping_rounds=40)
        onx = to_onnx(clr, X_train[:1].astype(numpy.float32),
                      options={XGBClassifier: {'zipmap': False}})
        sess = OnnxInference(onx)
        predict_list = [1., 20., 466., 0.]
        predict_array = numpy.array(predict_list).reshape(
            (1, -1)).astype(numpy.float32)
        pred_onx = sess.run({'X': predict_array})
        pred_onx = pred_onx['probabilities']
        pred_xgboost = clr.predict_proba(predict_array)
        self.assertEqualArray(pred_xgboost, pred_onx)


if __name__ == "__main__":
    unittest.main()
