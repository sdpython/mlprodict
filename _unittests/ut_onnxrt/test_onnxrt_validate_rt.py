"""
@brief      test log(time=5s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import (
    sklearn_operators, enumerate_validated_operator_opsets, summary_report
)


class TestOnnxrtValidateRt(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_pyrt_ort(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_pyrt_ort")
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LinearRegression"},
            fLOG=fLOG,
            runtime=['python', 'onnxruntime1'], debug=False,
            filter_exp=lambda m, p: '-64' not in p,
            benchmark=True, n_features=[None, 10]))

        self.assertGreater(len(rows), 1)
        df = DataFrame(rows)
        self.assertGreater(df.shape[1], 1)
        fLOG("output results")
        df.to_csv(os.path.join(temp, "sklearn_opsets_report.csv"), index=False)
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)
        piv = summary_report(df)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)
        rts = set(piv['runtime'])
        self.assertEqual(rts, {'python', 'onnxruntime1'})
        nfs = set(piv['n_features'])
        self.assertEqual(nfs, {4, 10})


if __name__ == "__main__":
    unittest.main()
