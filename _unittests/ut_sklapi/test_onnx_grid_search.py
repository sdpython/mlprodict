"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from pyquickhelper.pycode import ExtTestCase
from mlprodict.sklapi import OnnxTransformer


class TestInferenceSessionSklearnGridSearch(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_pipeline(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pca = PCA(n_components=2)
        pca.fit(X)

        onx = convert_sklearn(pca, initial_types=[
                              ('input', FloatTensorType((None, X.shape[1])))])
        onx_bytes = onx.SerializeToString()
        tr = OnnxTransformer(onx_bytes)

        pipe = make_pipeline(tr, LogisticRegression())
        pipe.fit(X, y)
        pred = pipe.predict(X)
        self.assertEqual(pred.shape, (150, ))
        skl_pred = pca.transform(X)
        skl_onx = pipe.steps[0][1].transform(X)
        self.assertEqualArray(skl_pred, skl_onx, decimal=5)

    def test_grid_search(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pca = PCA(n_components=2)
        pca.fit(X_train)
        onx = convert_sklearn(pca, initial_types=[
                              ('input', FloatTensorType((None, X.shape[1])))])
        onx_bytes = onx.SerializeToString()
        tr = OnnxTransformer(onx_bytes)

        pipe = make_pipeline(tr, LogisticRegression(solver='liblinear'))

        param_grid = [{'logisticregression__penalty': ['l2', 'l1']}]

        clf = GridSearchCV(pipe, param_grid, cv=3)
        clf.fit(X_train, y_train)
        bp = clf.best_params_
        self.assertIn(bp, ({'logisticregression__penalty': 'l1'},
                           {'logisticregression__penalty': 'l2'}))

        tr2 = OnnxTransformer(onx_bytes)
        tr2.fit()
        self.assertEqualArray(tr2.transform(X_test),
                              clf.best_estimator_.steps[0][1].transform(X_test))
        y_true, y_pred = y_test, clf.predict(X_test)
        cl = classification_report(y_true, y_pred)
        self.assertIn('precision', cl)
        sc = clf.score(X_test, y_test)
        self.assertGreater(sc, 0.70)

    def test_grid_search_onnx(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pca = PCA(n_components=2)
        pca.fit(X_train)
        onx = convert_sklearn(pca, initial_types=[
                              ('input', FloatTensorType((None, X.shape[1])))])
        onx_bytes2 = onx.SerializeToString()

        pca = PCA(n_components=3)
        pca.fit(X_train)
        onx = convert_sklearn(pca, initial_types=[
                              ('input', FloatTensorType((None, X.shape[1])))])
        onx_bytes3 = onx.SerializeToString()

        pipe = make_pipeline(OnnxTransformer(onx_bytes2),
                             LogisticRegression())

        param_grid = [{'onnxtransformer__onnx_bytes':
                       [onx_bytes2, onx_bytes3]}]

        clf = GridSearchCV(pipe, param_grid, cv=3)
        clf.fit(X_train, y_train)
        bp = clf.best_params_
        self.assertIn("onnxtransformer__onnx_bytes", bp)

        y_true, y_pred = y_test, clf.predict(X_test)
        cl = classification_report(y_true, y_pred)
        self.assertIn('precision', cl)
        sc = clf.score(X_test, y_test)
        self.assertGreater(sc, 0.70)


if __name__ == '__main__':
    unittest.main()
