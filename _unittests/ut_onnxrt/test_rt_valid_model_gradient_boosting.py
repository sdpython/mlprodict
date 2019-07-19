"""
@brief      test log(time=218s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import sklearn_operators, enumerate_validated_operator_opsets, summary_report


class TestRtValidateGradientBoosting(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 3)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_GradientBoostingRegressor1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GradientBoostingRegressor"}, opset_min=10, fLOG=fLOG,
            runtime='onnxruntime1', debug=False))
        self.assertGreater(len(rows), 2)
        df = DataFrame(rows)
        self.assertIn("max_rel_diff_batch", df.columns)
        self.assertGreater(df.shape[0], 1)
        piv = summary_report(df)
        self.assertGreater(piv.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
