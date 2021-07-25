"""
@brief      test log(time=3s)
"""
import sys
import unittest
from logging import getLogger
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateLightGbm(ExtTestCase):

    @skipif_circleci('too long')
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @unittest.skipIf(sys.platform == 'darwin', reason="stuck")
    def test_rt_LGBMClassifier_onnxruntime1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LGBMClassifier"},
            fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, p: '-64' not in p))
        self.assertGreater(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
