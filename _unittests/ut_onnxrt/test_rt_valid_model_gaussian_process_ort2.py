"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
from onnxruntime import __version__ as ort_version
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


threshold = "0.4.0"


class TestRtValidateGaussianProcessOrt2(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_circleci("to investigate, shape of predictions are different")
    @unittest.skipIf(compare_module_version(ort_version, threshold) <= 0,
                     reason="Node:Scan1 Field 'shape' of type is required but missing.")
    def test_rt_GaussianProcessRegressor_debug_std(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 4

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=debug,
            filter_exp=lambda m, s: "b-reg-std-NSV" in s))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
