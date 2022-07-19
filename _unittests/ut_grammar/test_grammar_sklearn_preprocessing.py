"""
@brief      test log(time=2s)
"""
import unittest
import platform
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing import check_model_representation
from mlprodict.grammar.cc.c_compilation import CompilationError


class TestGrammarSklearnPreprocessing(ExtTestCase):

    @unittest.skipIf(platform.system().lower() == "darwin",
                     reason="compilation issue with CFFI")
    def test_sklearn_scaler(self):
        from sklearn.preprocessing import StandardScaler
        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]],
                           dtype=numpy.float32)
        try:
            check_model_representation(
                StandardScaler, data, verbose=False)
        except (CompilationError, RuntimeError) as e:
            if "Visual Studio is not installed" in str(e):
                return
            raise AssertionError(  # pylint: disable=W0707
                f"Issue type {type(e)!r} exc {e!r}.")
        # The second compilation fails if suffix is not specified.
        check_model_representation(
            model=StandardScaler, X=data, verbose=False, suffix="_2")


if __name__ == "__main__":
    unittest.main()
