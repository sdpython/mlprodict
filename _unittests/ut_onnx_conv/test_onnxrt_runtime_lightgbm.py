"""
@brief      test log(time=3s)
"""
import sys
import unittest
from logging import getLogger
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType,
    BooleanTensorType)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx


class TestOnnxrtRuntimeLightGbm(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_missing(self):
        from mlprodict.onnx_conv.parsers.parse_lightgbm import WrappedLightGbmBooster

        r = WrappedLightGbmBooster._generate_classes(  # pylint: disable=W0212
            dict(num_class=1))
        self.assertEqual(r.tolist(), [0, 1])
        r = WrappedLightGbmBooster._generate_classes(  # pylint: disable=W0212
            dict(num_class=3))
        self.assertEqual(r.tolist(), [0, 1, 2])

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_onnxrt_python_lightgbm_categorical(self):
        from lightgbm import LGBMClassifier

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
        exp = gbm0.predict(X_test, raw_scores=False)
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
        exp = gbm0.predict_proba(X_test, raw_scores=False)
        model_def = to_onnx(gbm0, X)
        self.assertIn('ZipMap', str(model_def))

        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)),
                         ['output_label', 'output_probability'])
        df = pandas.DataFrame(y['output_probability'])
        self.assertEqual(df.shape, (X_test.shape[0], 2))
        self.assertEqual(exp.shape, (X_test.shape[0], 2))
        # self.assertEqualArray(exp, df.values, decimal=6)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_onnxrt_python_lightgbm_categorical3(self):
        from lightgbm import LGBMClassifier

        X = pandas.DataFrame({"A": numpy.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                              # int
                              "B": numpy.random.permutation([1, 2, 3] * 100),
                              # float
                              "C": numpy.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),
                              # bool
                              "D": numpy.random.permutation([True, False] * 150),
                              "E": pandas.Categorical(numpy.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                                      ordered=True)})  # str and ordered categorical
        y = numpy.random.permutation([0, 1, 2] * 100)
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
        exp = gbm0.predict(X_test, raw_scores=False)
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
        exp = gbm0.predict_proba(X_test, raw_scores=False)
        model_def = to_onnx(gbm0, X)
        self.assertIn('ZipMap', str(model_def))

        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)),
                         ['output_label', 'output_probability'])
        df = pandas.DataFrame(y['output_probability'])
        self.assertEqual(df.shape, (X_test.shape[0], 3))
        self.assertEqual(exp.shape, (X_test.shape[0], 3))
        # self.assertEqualArray(exp, df.values, decimal=6)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_onnxrt_python_lightgbm_categorical_iris(self):
        from lightgbm import LGBMClassifier, Dataset, train as lgb_train

        iris = load_iris()
        X, y = iris.data, iris.target
        X = (X * 10).astype(numpy.int32)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11)
        other_x = numpy.random.randint(
            0, high=10, size=(1500, X_train.shape[1]))
        X_train = numpy.vstack([X_train, other_x]).astype(dtype=numpy.int32)
        y_train = numpy.hstack(
            [y_train, numpy.zeros(500) + 3, numpy.zeros(500) + 4,
             numpy.zeros(500) + 5]).astype(dtype=numpy.int32)
        self.assertEqual(y_train.shape, (X_train.shape[0], ))
        y_train = y_train % 2

        # Classic
        gbm = LGBMClassifier()
        gbm.fit(X_train, y_train)
        exp = gbm.predict_proba(X_test)
        onx = to_onnx(gbm, initial_types=[
            ('X', Int64TensorType([None, X_train.shape[1]]))])
        self.assertIn('ZipMap', str(onx))
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
            "n_estimators": 2,
            "objective": "binary",
            "max_bin": 5,
            "min_child_samples": 100,
            'verbose': -1,
        }

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, initial_types=[
            ('X', Int64TensorType([None, X_train.shape[1]]))])
        self.assertIn('ZipMap', str(onx))
        oif = OnnxInference(onx)
        got = oif.run({'X': X_test})
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_onnxrt_python_lightgbm_categorical_iris_dataframe(self):
        from lightgbm import Dataset, train as lgb_train

        iris = load_iris()
        X, y = iris.data, iris.target
        X = (X * 10).astype(numpy.int32)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11)
        other_x = numpy.random.randint(
            0, high=10, size=(1500, X_train.shape[1]))
        X_train = numpy.vstack([X_train, other_x]).astype(dtype=numpy.int32)
        y_train = numpy.hstack(
            [y_train, numpy.zeros(500) + 3, numpy.zeros(500) + 4,
             numpy.zeros(500) + 5]).astype(dtype=numpy.int32)
        self.assertEqual(y_train.shape, (X_train.shape[0], ))
        y_train = y_train % 2

        df_train = pandas.DataFrame(X_train)
        df_train.columns = ['c1', 'c2', 'c3', 'c4']
        df_train['c1'] = df_train['c1'].astype('category')
        df_train['c2'] = df_train['c2'].astype('category')
        df_train['c3'] = df_train['c3'].astype('category')
        df_train['c4'] = df_train['c4'].astype('category')

        df_test = pandas.DataFrame(X_test)
        df_test.columns = ['c1', 'c2', 'c3', 'c4']
        df_test['c1'] = df_test['c1'].astype('category')
        df_test['c2'] = df_test['c2'].astype('category')
        df_test['c3'] = df_test['c3'].astype('category')
        df_test['c4'] = df_test['c4'].astype('category')

        # categorical_feature=[0, 1]
        train_data = Dataset(
            df_train, label=y_train)

        params = {
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 2,
            "objective": "binary",
            "max_bin": 5,
            "min_child_samples": 100,
            'verbose': -1,
        }

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, df_train)
        self.assertIn('ZipMap', str(onx))

        oif = OnnxInference(onx)
        got = oif.run(df_test)
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

        onx.ir_version = get_ir_version_from_onnx()
        oif = OnnxInference(onx, runtime='onnxruntime1')
        got = oif.run(df_test)
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

        onx = to_onnx(booster, df_train,
                      options={booster.__class__: {'cast': True}})
        self.assertIn('op_type: "Cast"', str(onx))
        oif = OnnxInference(onx)
        got = oif.run(df_test)
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_lightgbm_booster_classifier(self):
        from lightgbm import Dataset, train as lgb_train

        X = numpy.array([[0, 1], [1, 1], [2, 0], [1, 2]], dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = Dataset(X, label=y)
        model = lgb_train({'boosting_type': 'rf', 'objective': 'binary',
                           'n_estimators': 3, 'min_child_samples': 1,
                           'subsample_freq': 1, 'bagging_fraction': 0.5,
                           'feature_fraction': 0.5},
                          data)
        model_onnx = to_onnx(model, X)
        self.assertNotEmpty(model_onnx)


if __name__ == "__main__":
    unittest.main()
