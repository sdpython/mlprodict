"""
@brief      test log(time=2s)
"""
import unittest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import ignore_warnings
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.optim.sklearn_helper import enumerate_pipeline_models, inspect_sklearn_model


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

    @ignore_warnings(category=(UserWarning, RuntimeWarning, DeprecationWarning))
    def test_statistics_lin(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression()
        clr.fit(X_train, y_train)
        res = inspect_sklearn_model(clr)
        self.assertEqual(res, {'ncoef': 3, 'nlin': 1, 'nop': 1})


if __name__ == "__main__":
    unittest.main()
