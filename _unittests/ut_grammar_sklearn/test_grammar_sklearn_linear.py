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


from src.mlprodict.testing import iris_data, check_model_representation
from src.mlprodict.grammar_sklearn import sklearn2graph, identify_interpreter
from src.mlprodict.cc import compile_c_function


class TestGrammarSklearnLinear(ExtTestCase):

    def test_sklearn_lr(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        gr = identify_interpreter(lr)
        self.assertCallable(gr)

    def test_sklearn_train_lr(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        lr = LogisticRegression()
        lr.fit(X, y)
        gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

        X = numpy.array([[numpy.float32(1), numpy.float32(2)]])
        e1 = lr.predict(X)
        p1 = lr.decision_function(X)
        e2 = gr.execute(Features=X[0, :])
        self.assertEqual(e1[0], e2[0])
        self.assertEqualFloat(p1, e2[1])

        ser = gr.export(lang="json", hook={'array': lambda v: v.tolist()})
        self.maxDiff = None
        self.assertEqual(6, len(ser))
        # import json
        # print(json.dumps(ser, sort_keys=True, indent=2))
        # self.assertEqual(ser, exp) # training not always the same

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
        gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

        code_c = gr.export(lang="c")['code']
        if code_c is None:
            raise ValueError("cannot be None")

        X = numpy.array([[numpy.float32(1), numpy.float32(2)]])
        fct = compile_c_function(code_c, 2)

        e2 = fct(X[0, :])
        e1 = lr.predict(X)
        p1 = lr.decision_function(X)
        self.assertEqual(e1[0], e2[0])
        self.assertEqualFloat(p1, e2[1])

    def test_sklearn_linear_regression(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.linear_model import LinearRegression
        X, y = iris_data()
        check_model_representation(
            LinearRegression, X, y, verbose=False, fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
