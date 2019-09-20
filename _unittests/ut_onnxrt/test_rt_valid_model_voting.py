"""
@brief      test log(time=4s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import (
    sklearn_operators, enumerate_validated_operator_opsets,
    summary_report
)


class TestRtValidateVoting(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_Voting(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"VotingRegressor", "VotingClassifier",
                             'LinearRegression'},
            opset_min=9, fLOG=fLOG,
            runtime='python', debug=False))
        self.assertGreater(len(rows), 4)
        df = DataFrame(rows)
        piv = summary_report(df)
        reg = piv[piv.name == 'VotingRegressor']
        self.assertGreater(reg.shape[0], 1)
        nonan = reg['opset10'].dropna()
        if nonan.shape[0] == 4:
            self.assertEqual(nonan.shape[0], reg.shape[0])
        else:
            self.assertGreater(nonan.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
