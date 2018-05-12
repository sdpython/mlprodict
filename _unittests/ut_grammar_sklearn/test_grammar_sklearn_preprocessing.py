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

from src.mlprodict.testing import check_model_representation


class TestGrammarSklearnPreprocessing(ExtTestCase):

    def test_sklearn_scaler(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        from sklearn.preprocessing import StandardScaler
        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]],
                           dtype=numpy.float32)
        check_model_representation(
            StandardScaler, data, verbose=False, fLOG=fLOG)
        # The second compilation fails if suffix is not specified.
        check_model_representation(
            model=StandardScaler, X=data, verbose=False, fLOG=fLOG, suffix="_2")


if __name__ == "__main__":
    unittest.main()
