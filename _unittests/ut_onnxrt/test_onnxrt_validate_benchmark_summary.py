"""
@brief      test log(time=6s)
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
    enumerate_validated_operator_opsets, summary_report,
    get_opset_number_from_onnx
)


class TestOnnxrtValidateBenchmarkSummary(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_benchmark_errros(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_benchmark_summary")
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"RFE", "DecisionTreeRegressor"}, opset_min=10,
            benchmark=True, fLOG=fLOG))
        self.assertGreater(len(rows), 1)
        df = DataFrame(rows)
        for col in ['skl', 'batch']:
            self.assertIn('lambda-' + col, df.columns)
        for col in ['1', '10']:
            self.assertIn('time-ratio-N=' + col, df.columns)
        self.assertGreater(df.shape[1], 1)
        self.assertGreater(df.loc[0, "tostring_time"], 0)
        piv = summary_report(df)
        self.assertGreater(piv.shape[1], 1)
        self.assertIn('RT/SKL-N=1', piv.columns)
        self.assertNotIn('RT/SKL-N=10', piv.columns)
        self.assertIn('N=10', piv.columns)
        fLOG("output results")
        ops = 'opset%d' % get_opset_number_from_onnx()
        li = len(piv[ops].notnull())
        self.assertEqual(li, piv.shape[0])
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)


if __name__ == "__main__":
    unittest.main()
