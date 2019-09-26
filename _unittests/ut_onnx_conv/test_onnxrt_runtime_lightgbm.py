"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from lightgbm import LGBMClassifier, Dataset, train as lgb_train
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import skl2onnx
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType,
    BooleanTensorType, Int32TensorType
)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx


class TestOnnxrtRuntimeLightGbm(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxrt_python_lightgbm_categorical(self):

        X = pandas.DataFrame({"A": numpy.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                              # int
                              "B": numpy.random.permutation([1, 2, 3] * 100),
                              # float
                              "C": numpy.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),
                              # bool
                              "D": numpy.random.permutation([True, False] * 150),
                              "E": pandas.Categorical(numpy.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                                      ordered=True)})  # str and ordered categorical
        y = numpy.random.permutation([0, 1] * 150)
        X_test = pandas.DataFrame({"A": numpy.random.permutation(['a', 'b', 'e'] * 20),  # unseen category
                                   "B": numpy.random.permutation([1, 3] * 30),
                                   "C": numpy.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                                   "D": numpy.random.permutation([True, False] * 30),
                                   "E": pandas.Categorical(numpy.random.permutation(['z', 'y'] * 30),
                                                           ordered=True)})
        cat_cols_actual = ["A", "B", "C", "D"]
        X[cat_cols_actual] = X[cat_cols_actual].astype('category')
        X_test[cat_cols_actual] = X_test[cat_cols_actual].astype('category')
        gbm0 = LGBMClassifier().fit(X, y)
        exp = gbm0.predict(X_test, raw_score=False)
        self.assertNotEmpty(exp)

        init_types = [('A', StringTensorType()),
                      ('B', Int64TensorType()),
                      ('C', FloatTensorType()),
                      ('D', BooleanTensorType()),
                      ('E', StringTensorType())]
        self.assertRaise(lambda: to_onnx(gbm0, initial_types=init_types), RuntimeError,
                         "at most 1 input(s) is(are) supported")

        X = X[['C']].values.astype(numpy.float32)
        X_test = X_test[['C']].values.astype(numpy.float32)
        gbm0 = LGBMClassifier().fit(X, y, categorical_feature=[0])
        exp = gbm0.predict_proba(X_test, raw_score=False)
        model_def = to_onnx(gbm0, X)

        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)),
                         ['output_label', 'output_probability'])
        df = pandas.DataFrame(y['output_probability'])
        self.assertEqual(df.shape, (X_test.shape[0], 2))
        self.assertEqual(exp.shape, (X_test.shape[0], 2))
        # self.assertEqualArray(exp, df.values, decimal=6)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxrt_python_lightgbm_categorical_iris(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X = (X * 10).astype(numpy.int32)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11)
        other_x = numpy.random.randint(
            0, high=10000, size=(1000, X_train.shape[1]))
        X_train = numpy.vstack([X_train, other_x]).astype(dtype=numpy.int32)
        y_train = numpy.hstack(
            [y_train, numpy.zeros(1000) + 4]).astype(dtype=numpy.int32)

        # Classic
        gbm = LGBMClassifier()
        gbm.fit(X_train, y_train)
        exp = gbm.predict_proba(X_test)
        onx = to_onnx(gbm, initial_types=[
            ('X', Int32TensorType([None, X_train.shape[1]]))])
        oif = OnnxInference(onx)
        got = oif.run({'X': X_test})
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values, decimal=5)

        # categorical_feature=[0, 1]
        train_data = Dataset(
            X_train, label=y_train,
            feature_name=['c1', 'c2', 'c3', 'c4'],
            categorical_feature=['c1', 'c2'])

        params = {
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 100,
            "objective": "binary",
            "max_bin": 10,
            "min_child_samples": 100,
            'verbose': -1,
        }

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, initial_types=[
            ('X', Int32TensorType([None, X_train.shape[1]]))])
        oif = OnnxInference(onx)
        got = oif.run({'X': X_test})
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)


if __name__ == "__main__":
    unittest.main()
