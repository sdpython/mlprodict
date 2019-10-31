"""
@brief      test log(time=9s)
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


class TestRtValidateCalibrated(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_CalibratedClassifierCV_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"CalibratedClassifierCV"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=True,
            filter_scenario=lambda m, p, sc, ex, ex2: 'sgd' in sc))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)
        maxv = max(row['max_rel_diff_batch'] for row in rows)
        self.assertLesser(maxv, 1e-5)


if __name__ == "__main__":
    unittest.main()
