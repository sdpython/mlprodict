"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing import iris_data, check_model_representation


class TestGrammarSklearnTree(ExtTestCase):

    def test_sklearn_tree1(self):
        from sklearn.tree import DecisionTreeRegressor
        X, y = iris_data()
        check_model_representation(DecisionTreeRegressor(max_depth=1), X, y,
                                   verbose=False, suffix="t1")

    def test_sklearn_tree2(self):
        from sklearn.tree import DecisionTreeRegressor
        X, y = iris_data()
        check_model_representation(model=DecisionTreeRegressor(max_depth=2),
                                   X=X, y=y, verbose=True, suffix="t2")


if __name__ == "__main__":
    unittest.main()
