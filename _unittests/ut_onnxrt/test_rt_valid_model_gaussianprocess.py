"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateGaussianProcess(ExtTestCase):

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_onnxruntime_nofit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime', debug=True, filter_exp=lambda s: "nofit-std" in s))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_python_nofit(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=True, filter_exp=lambda s: "nofit" in s))
        self.assertGreater(len(rows), 6)
        self.assertGreater(len(buffer), 1)


if __name__ == "__main__":
    unittest.main()
