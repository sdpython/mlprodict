"""
@brief      test log(time=65s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import (
    get_temp_folder, ExtTestCase, skipif_circleci, skipif_appveyor)
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report


class TestOnnxrtValidateBenchmark(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_benchmark(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_benchmark")
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LinearRegression"}, opset_min=10,
            benchmark=True, fLOG=fLOG, time_kwargs_fact='lin'))
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
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)

    @skipif_circleci('too long')
    @skipif_appveyor('crashes')
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_benchmark_all(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 11 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_benchmark_all")
        rows = []
        for row in enumerate_validated_operator_opsets(
                verbose, opset_min=10, benchmark=True,
                fLOG=fLOG, runtime="onnxruntime1",
                versions=True):
            rows.append(row)
            if len(rows) > 6:
                break
            fLOG('i')
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
        self.assertIn('v_numpy', df.columns)
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)
        self.assertIn('v_numpy', piv.columns)


if __name__ == "__main__":
    unittest.main()
