"""
@brief      test log(time=400s)
"""
import sys
import unittest
from logging import getLogger
import numpy
import pandas
from onnxruntime import InferenceSession
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType,
    BooleanTensorType, DoubleTensorType)
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict import __max_supported_opsets__ as TARGET_OPSET, get_ir_version


class TestOnnxrtRuntimeLightGbm(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_missing(self):
        from mlprodict.onnx_conv.operator_converters.parse_lightgbm import (
            WrappedLightGbmBooster)

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

        X = pandas.DataFrame(
            {"A": numpy.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
             # int
             "B": numpy.random.permutation([1, 2, 3] * 100),
             # float
             "C": numpy.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),
             # bool
             "D": numpy.random.permutation([True, False] * 150),
             "E": pandas.Categorical(numpy.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                     ordered=True)})  # str and ordered categorical
        y = numpy.random.permutation([0, 1] * 150)
        X_test = pandas.DataFrame(
            {"A": numpy.random.permutation(['a', 'b', 'e'] * 20),  # unseen category
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

        init_types = [('A', StringTensorType()), ('B', Int64TensorType()),
                      ('C', FloatTensorType()), ('D', BooleanTensorType()),
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

        X = pandas.DataFrame(
            {"A": numpy.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
             # int
             "B": numpy.random.permutation([1, 2, 3] * 100),
             # float
             "C": numpy.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),
             # bool
             "D": numpy.random.permutation([True, False] * 150),
             "E": pandas.Categorical(numpy.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                     ordered=True)})  # str and ordered categorical
        y = numpy.random.permutation([0, 1, 2] * 100)
        X_test = pandas.DataFrame(
            {"A": numpy.random.permutation(['a', 'b', 'e'] * 20),  # unseen category
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
        model_def = to_onnx(gbm0, X, target_opset=TARGET_OPSET)
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
            ('X', Int64TensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET)
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
            "boosting_type": "gbdt", "learning_rate": 0.05,
            "n_estimators": 2, "objective": "binary",
            "max_bin": 5, "min_child_samples": 100,
            'verbose': -1}

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, initial_types=[
            ('X', Int64TensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIn('ZipMap', str(onx))
        oif = OnnxInference(onx)
        got = oif.run({'X': X_test})
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_onnxrt_python_lightgbm_categorical_iris_booster3(self):
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

        # Classic
        gbm = LGBMClassifier()
        gbm.fit(X_train, y_train)
        exp = gbm.predict_proba(X_test)
        onx = to_onnx(gbm, initial_types=[
            ('X', Int64TensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET)
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
            "boosting_type": "gbdt", "learning_rate": 0.05,
            "n_estimators": 2, "objective": "binary",
            "max_bin": 5, "min_child_samples": 100,
            'verbose': -1}

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, initial_types=[
            ('X', Int64TensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIn('ZipMap', str(onx))
        oif = OnnxInference(onx)
        got = oif.run({'X': X_test})
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_onnxrt_python_lightgbm_categorical_iris_booster3_real(self):
        from lightgbm import LGBMClassifier, Dataset, train as lgb_train

        iris = load_iris()
        X, y = iris.data, iris.target
        X = (X * 10).astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11)

        # Classic
        gbm = LGBMClassifier()
        gbm.fit(X_train, y_train)
        exp = gbm.predict_proba(X_test)
        onx = to_onnx(gbm.booster_, initial_types=[
            ('X', FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET)
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
            "boosting_type": "gbdt", "learning_rate": 0.05,
            "n_estimators": 2, "objective": "multiclass",
            "max_bin": 5, "min_child_samples": 100,
            'verbose': -1, 'num_class': 3}

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, initial_types=[
            ('X', FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIn('ZipMap', str(onx))
        oif = OnnxInference(onx)
        got = oif.run({'X': X_test})
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values, decimal=5)

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
            "boosting_type": "gbdt", "learning_rate": 0.05,
            "n_estimators": 2, "objective": "binary",
            "max_bin": 5, "min_child_samples": 100,
            'verbose': -1}

        booster = lgb_train(params, train_data)
        exp = booster.predict(X_test)

        onx = to_onnx(booster, df_train, target_opset=TARGET_OPSET)
        self.assertIn('ZipMap', str(onx))

        oif = OnnxInference(onx)
        got = oif.run(df_test)
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

        onx.ir_version = get_ir_version(TARGET_OPSET)
        oif = OnnxInference(onx, runtime='onnxruntime1')
        got = oif.run(df_test)
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

        onx = to_onnx(booster, df_train,
                      options={booster.__class__: {'cast': True}},
                      target_opset=TARGET_OPSET)
        self.assertIn('op_type: "Cast"', str(onx))
        oif = OnnxInference(onx)
        got = oif.run(df_test)
        values = pandas.DataFrame(got['output_probability']).values
        self.assertEqualArray(exp, values[:, 1], decimal=5)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_lightgbm_booster_classifier(self):
        from lightgbm import Dataset, train as lgb_train

        X = numpy.array([[0, 1], [1, 1], [2, 0], [1, 2]], dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = Dataset(X, label=y)
        model = lgb_train({'boosting_type': 'rf', 'objective': 'binary',
                           'n_estimators': 3, 'min_child_samples': 1,
                           'subsample_freq': 1, 'bagging_fraction': 0.5,
                           'feature_fraction': 0.5, 'average_output': True},
                          data)
        model_onnx = to_onnx(model, X, verbose=0, rewrite_ops=True,
                             target_opset=TARGET_OPSET)
        self.assertNotEmpty(model_onnx)

    # missing values

    @staticmethod
    def _predict_with_onnx(model, X):
        session = InferenceSession(model.SerializeToString())
        output_names = [s_output.name for s_output in session.get_outputs()]
        input_names = [s_input.name for s_input in session.get_inputs()]
        if len(input_names) > 1:
            raise RuntimeError(
                "Test expects one input. Found multiple inputs: %r."
                "" % input_names)
        input_name = input_names[0]
        return session.run(output_names, {input_name: X})[0][:, 0]

    def _assert_almost_equal(self, actual, desired, decimal=7, frac=1.0, msg=""):
        self.assertGreater(frac, 0)
        self.assertLesser(frac, 1)
        success_abs = (abs(actual - desired) <= (10 ** -decimal)).sum()
        success_rel = success_abs / len(actual)
        if success_abs == 0:
            raise AssertionError(
                "Wrong conversion. %s\n-----\n%r\n------\n%r"
                "" % (msg, desired[:5], actual[:5]))
        self.assertGreater(success_rel, frac)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_missing_values(self):
        from lightgbm import LGBMRegressor

        _N_DECIMALS = 5
        _FRAC = 0.9999

        _y = numpy.array([0, 0, 1, 1, 1])
        _X_train = numpy.array([[1.0, 0.0], [1.0, -1.0], [1.0, -1.0],
                                [2.0, -1.0], [2.0, -1.0]],
                               dtype=numpy.float32)
        _X_test = numpy.array([[1.0, numpy.nan]], dtype=numpy.float32)

        _INITIAL_TYPES = [
            ("input", FloatTensorType([None, _X_train.shape[1]]))]

        regressor = LGBMRegressor(
            objective="regression", min_data_in_bin=1, min_data_in_leaf=1,
            n_estimators=1, learning_rate=1)
        regressor.fit(_X_train, _y)
        regressor_onnx = to_onnx(
            regressor, initial_types=_INITIAL_TYPES, rewrite_ops=True,
            target_opset=TARGET_OPSET)
        y_pred = regressor.predict(_X_test)
        y_pred_onnx = self._predict_with_onnx(regressor_onnx, _X_test)
        self._assert_almost_equal(
            y_pred, y_pred_onnx, decimal=_N_DECIMALS, frac=_FRAC,
            msg="Missing values.")

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_missing_values_rf(self):
        from lightgbm import LGBMRegressor

        _N_DECIMALS = 5
        _FRAC = 0.9999

        _y = numpy.array([0, 0, 1, 1, 1])
        _X_train = numpy.array([[1.0, 0.0], [1.0, -1.0], [1.0, -1.0],
                                [2.0, -1.0], [2.0, -1.0]],
                               dtype=numpy.float32)
        _X_test = numpy.array([[1.0, numpy.nan]], dtype=numpy.float32)

        _INITIAL_TYPES = [
            ("input", FloatTensorType([None, _X_train.shape[1]]))]

        regressor = LGBMRegressor(
            objective="regression", boosting_type='rf',
            n_estimators=10, bagging_freq=1, bagging_fraction=0.5)
        regressor.fit(_X_train, _y)
        regressor_onnx = to_onnx(
            regressor, initial_types=_INITIAL_TYPES, rewrite_ops=True,
            target_opset=TARGET_OPSET)
        y_pred = regressor.predict(_X_test)
        y_pred_onnx = self._predict_with_onnx(regressor_onnx, _X_test)
        self._assert_almost_equal(
            y_pred, y_pred_onnx, decimal=_N_DECIMALS, frac=_FRAC,
            msg="Missing values.")

    # objectives

    @staticmethod
    def _calc_initial_types(X):
        _DTYPE_MAP = {"float64": DoubleTensorType,
                      "float32": FloatTensorType}

        dtypes = set(str(dtype) for dtype in X.dtypes)
        if len(dtypes) > 1:
            raise RuntimeError(
                "Test expects homogenous input matrix. Found multiple dtypes: %r." % dtypes)
        dtype = dtypes.pop()
        tensor_type = _DTYPE_MAP[dtype]
        return [("input", tensor_type(X.shape))]

    @staticmethod
    def _predict_with_onnx(model, X):
        session = InferenceSession(model.SerializeToString())
        output_names = [s_output.name for s_output in session.get_outputs()]
        input_names = [s_input.name for s_input in session.get_inputs()]
        if len(input_names) > 1:
            raise RuntimeError(
                "Test expects one input. Found multiple inputs: %r." % input_names)
        input_name = input_names[0]
        if hasattr(X, "values"):
            return session.run(output_names, {input_name: X.values})[0][:, 0]
        return session.run(output_names, {input_name: X})[0][:, 0]

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_objective(self):
        from lightgbm import LGBMRegressor

        _N_ROWS = 10000
        _N_COLS = 10
        _N_DECIMALS = 5
        _FRAC = 0.9997

        _X = pandas.DataFrame(numpy.random.random(
            size=(_N_ROWS, _N_COLS)).astype(numpy.float32))
        _Y = pandas.Series(numpy.random.random(size=_N_ROWS))

        _objectives = ("regression", "poisson", "gamma")

        for objective in _objectives:
            with self.subTest(X=_X, objective=objective):
                initial_types = self._calc_initial_types(_X)
                regressor = LGBMRegressor(objective=objective)
                regressor.fit(_X, _Y)
                regressor_onnx = to_onnx(
                    regressor, initial_types=initial_types,
                    rewrite_ops=True, target_opset=TARGET_OPSET)
                y_pred = regressor.predict(_X)
                y_pred_onnx = self._predict_with_onnx(regressor_onnx, _X)
                self._assert_almost_equal(
                    y_pred, y_pred_onnx, decimal=_N_DECIMALS, frac=_FRAC,
                    msg="Objective=%r" % objective)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_objective_boosting_rf(self):
        from lightgbm import LGBMRegressor

        _N_ROWS = 10000
        _N_COLS = 10
        _N_DECIMALS = 5
        _FRAC = 0.9997

        _X = pandas.DataFrame(numpy.random.random(
            size=(_N_ROWS, _N_COLS)).astype(numpy.float32))
        _Y = pandas.Series(numpy.random.random(size=_N_ROWS))

        _objectives = ("regression",)

        for objective in _objectives:
            with self.subTest(X=_X, objective=objective):
                initial_types = self._calc_initial_types(_X)
                regressor = LGBMRegressor(
                    objective=objective, boosting='rf', bagging_freq=3,
                    bagging_fraction=0.5, n_estimators=10)
                regressor.fit(_X, _Y)
                regressor_onnx = to_onnx(
                    regressor, initial_types=initial_types,
                    rewrite_ops=True, target_opset=TARGET_OPSET)
                y_pred = regressor.predict(_X)
                y_pred_onnx = self._predict_with_onnx(regressor_onnx, _X) / 10
                self._assert_almost_equal(
                    y_pred, y_pred_onnx, decimal=_N_DECIMALS, frac=_FRAC,
                    msg="Objective=%r" % objective)

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_lgbm_regressor10(self):
        from lightgbm import LGBMRegressor
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        reg = LGBMRegressor(max_depth=2, n_estimators=4, seed=0)
        reg.fit(X_train, y_train)
        expected = reg.predict(X_test)

        # float
        onx = to_onnx(reg, X_train, rewrite_ops=True)
        oinf = OnnxInference(onx)
        got1 = oinf.run({'X': X_test})['variable']

        # float split
        onx = to_onnx(reg, X_train, options={'split': 2},
                      rewrite_ops=True, target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got2 = oinf.run({'X': X_test})['variable']

        # final check
        self.assertEqualArray(expected, got1, decimal=5)
        self.assertEqualArray(expected, got2, decimal=5)

    @ignore_warnings((RuntimeWarning, UserWarning))
    def test_lgbm_regressor(self):
        from lightgbm import LGBMRegressor
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        reg = LGBMRegressor(max_depth=2, n_estimators=100, seed=0)
        reg.fit(X_train, y_train)
        expected = reg.predict(X_test)

        # double
        onx = to_onnx(reg, X_train.astype(numpy.float64),
                      rewrite_ops=True, target_opset=TARGET_OPSET)
        self.assertIn("TreeEnsembleRegressorDouble", str(onx))
        oinf = OnnxInference(onx)
        got0 = oinf.run(
            {'X': X_test.astype(numpy.float64)})['variable']
        self.assertEqualArray(expected, got0)

        # float
        onx = to_onnx(reg, X_train, rewrite_ops=True,
                      target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got1 = oinf.run({'X': X_test})['variable']
        self.assertEqualArray(expected, got1, decimal=5)

        # float split
        onx = to_onnx(reg, X_train, options={'split': 10},
                      rewrite_ops=True,
                      target_opset=TARGET_OPSET)
        oinf = OnnxInference(onx)
        got2 = oinf.run({'X': X_test})['variable']
        self.assertEqualArray(expected, got2, decimal=5)
        oinf = OnnxInference(onx, runtime='onnxruntime1')
        got3 = oinf.run({'X': X_test})['variable']
        self.assertEqualArray(expected, got3.ravel(), decimal=5)

        # final
        d0 = numpy.abs(expected.ravel() - got0).mean()
        d1 = numpy.abs(expected.ravel() - got1).mean()
        d2 = numpy.abs(expected.ravel() - got2).mean()
        self.assertGreater(d1, d0)
        self.assertGreater(d1, d2)


if __name__ == "__main__":
    unittest.main()
