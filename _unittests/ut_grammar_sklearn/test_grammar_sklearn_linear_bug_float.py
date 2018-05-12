"""
@brief      test log(time=2s)
"""
import sys
import os
import unittest
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src


from src.mlprodict.grammar_sklearn import sklearn2graph
from src.mlprodict.grammar.exc import Float32InfError


class TestGrammarSklearnLinearBugFloat(ExtTestCase):

    def test_sklearn_train_lr_into_c(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import load_iris
        import cffi
        fLOG("cffi", cffi.__version__)
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
