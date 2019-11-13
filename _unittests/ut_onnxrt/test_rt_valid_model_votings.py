"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import (
    enumerate_validated_operator_opsets, summary_report
)


class TestRtValidateVotingClassifier(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_VotingClassifier_onnxruntime1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"VotingClassifier"}, opset_min=9,
            opset_max=11, fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, p: 'm-cl' in p))
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[0])
        self.assertIn('onx_size', rows[-1])
        piv = summary_report(DataFrame(rows))
        self.assertGreater(piv.shape[0], 1)

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
        self.assertEqual(nonan.shape[0], reg.shape[0])


if __name__ == "__main__":
    unittest.main()
