"""
@brief      test log(time=218s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase, is_travis_or_appveyor
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import sklearn_operators, enumerate_validated_operator_opsets, summary_report


class TestOnnxrtValidateOnnxRuntimeWhole(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 3)

    def test_validate_GradientBoostingRegressor_whole(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GradientBoostingRegressor"}, opset_min=11, fLOG=fLOG,
            runtime='onnxruntime-whole', debug=False))
        self.assertIn(len(rows), (1, 2, 3))
        df = DataFrame(rows)
        self.assertIn("max_abs_diff_batch", df.columns)
        self.assertGreater(df.shape[0], 2)
        piv = summary_report(df)
        self.assertIn(piv.shape[0], (1, 2, 3))

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_all_onnxruntime_whole(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_all_onnxruntime_whole")
        if False:  # pylint: disable=W0125
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"GradientBoostingRegressor"}, opset_min=11, fLOG=fLOG,
                runtime='onnxruntime-whole', debug=True))
        else:
            rows = []
            for row in enumerate_validated_operator_opsets(verbose, debug=None, fLOG=fLOG,
                                                           runtime='onnxruntime-whole',
                                                           dump_folder=temp):
                rows.append(row)
                if is_travis_or_appveyor() and len(rows) > 20:
                    break

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


if __name__ == "__main__":
    unittest.main()
