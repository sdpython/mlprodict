"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxAdd
)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt.optim.sklearn_helper import enumerate_pipeline_models, inspect_sklearn_model
from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics
from mlprodict.onnx_conv import to_onnx
from mlprodict.tools import get_opset_number_from_onnx


class TestSklearnHelper(ExtTestCase):

    def test_pipeline(self):
        numeric_features = ['age', 'fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['embarked', 'sex', 'pclass']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ])

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(solver='lbfgs'))])
        steps = list(enumerate_pipeline_models(clf))
        self.assertEqual(len(steps), 9)

    def test_union_features(self):
        model = Pipeline([('scaler1', StandardScaler()),
                          ('union', FeatureUnion([
                              ('scaler2', StandardScaler()),
                              ('scaler3', MinMaxScaler())]))])
        steps = list(enumerate_pipeline_models(model))
        self.assertEqual(len(steps), 5)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_statistics_rf(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = RandomForestRegressor(n_estimators=10, n_jobs=1, max_depth=4)
        clr.fit(X_train, y_train)
        res = inspect_sklearn_model(clr)
        self.assertEqual(res['max_depth'], 4)
        self.assertEqual(res['ntrees'], 10)
        onx = to_onnx(clr, X_train[:1].astype(numpy.float32))
        ostats = onnx_statistics(onx)
        for k, v in {'nnodes': 1, 'doc_string': '', 'domain': 'ai.onnx', 'model_version': 0,
                     'producer_name': 'skl2onnx', 'ai.onnx.ml': 1}.items():
            self.assertEqual(ostats[k], v)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_statistics_adaboost(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = AdaBoostRegressor(n_estimators=10)
        clr.fit(X_train, y_train)
        res = inspect_sklearn_model(clr)
        self.assertEqual(res['max_depth'], 3)
        self.assertGreater(res['ntrees'], 1)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_statistics_pipeline_rf(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = Pipeline([('scaler1', StandardScaler()),
                        ('rf', RandomForestRegressor(n_estimators=10, n_jobs=1, max_depth=4))])
        clr.fit(X_train, y_train)
        res = inspect_sklearn_model(clr)
        self.assertEqual(res['max_depth'], 4)
        self.assertEqual(res['ntrees'], 10)
        self.assertEqual(res['nop'], 11)
        onx = to_onnx(clr, X_train[:1].astype(numpy.float32))
        ostats = onnx_statistics(onx)
        for k, v in {'nnodes': 2, 'doc_string': '', 'domain': 'ai.onnx', 'model_version': 0,
                     'producer_name': 'skl2onnx', 'ai.onnx.ml': 1}.items():
            self.assertEqual(ostats[k], v)

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_statistics_lin(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)
        res = inspect_sklearn_model(clr)
        self.assertEqual(res, {'ncoef': 3, 'nlin': 1, 'nop': 1})

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_statistics_pipeline_sgd(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SGDClassifier()
        clr.fit(X_train, y_train)
        onx = to_onnx(clr, X_train[:1].astype(numpy.float32))
        ostats = onnx_statistics(onx)
        for k, v in {'nnodes': 9, 'doc_string': '', 'domain': 'ai.onnx', 'model_version': 0,
                     'producer_name': 'skl2onnx', 'ai.onnx.ml': 1}.items():
            self.assertEqual(ostats[k], v)
        self.assertIn('', ostats)
        self.assertIn("op_Cast", ostats)

    def test_onnx_stat_recursive(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        cop = OnnxAdd(OnnxIdentity('input'), 'input')
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=get_opset_number_from_onnx())
        stats = onnx_statistics(model_def)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)


if __name__ == "__main__":
    unittest.main()
