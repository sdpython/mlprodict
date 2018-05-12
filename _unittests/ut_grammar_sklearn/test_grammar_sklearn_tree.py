"""
@brief      test log(time=3s)
"""

import sys
import os
import unittest
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


class TestGrammarSklearnTree(ExtTestCase):

    def test_sklearn_tree1(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.tree import DecisionTreeRegressor
        X, y = iris_data()
        check_model_representation(DecisionTreeRegressor(max_depth=1), X, y,
                                   verbose=False, suffix="t1", fLOG=fLOG)

    def test_sklearn_tree2(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.tree import DecisionTreeRegressor
        X, y = iris_data()
        check_model_representation(model=DecisionTreeRegressor(max_depth=2),
                                   X=X, y=y, verbose=True, suffix="t2")


if __name__ == "__main__":
    unittest.main()
