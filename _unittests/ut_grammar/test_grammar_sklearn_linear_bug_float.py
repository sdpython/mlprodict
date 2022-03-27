"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.grammar.grammar_sklearn import sklearn2graph
from mlprodict.grammar.grammar_sklearn.grammar.exc import Float32InfError


class TestGrammarSklearnLinearBugFloat(ExtTestCase):

    def test_sklearn_train_lr_into_c(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        lr = LogisticRegression()
        lr.fit(X, y)

        # We replace by double too big for floats.
        lr.coef_ = numpy.array([[2.45, -3e250]])
        self.assertRaise(lambda: sklearn2graph(
            lr, output_names=['Prediction', 'Score']), Float32InfError)


if __name__ == "__main__":
    unittest.main()
