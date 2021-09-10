"""
@brief      test log(time=16s)
"""
import unittest
from logging import getLogger
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateNormalizer(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_Normalizer_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"Normalizer"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_Normalizer_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"Normalizer"},
            fLOG=myprint,
            runtime='python', debug=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)


if __name__ == "__main__":
    unittest.main()
