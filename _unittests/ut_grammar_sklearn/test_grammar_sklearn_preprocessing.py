"""
@brief      test log(time=2s)
"""
import sys
import os
import unittest


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


from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from src.mlprodict.testing import check_model_representation


class TestGrammarSklearnPreprocessing(ExtTestCase):

    def test_sklearn_scaler(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.preprocessing import StandardScaler
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        check_model_representation(
            StandardScaler, data, verbose=False, fLOG=fLOG)
        check_model_representation(
            model=StandardScaler, verbose=False, fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
