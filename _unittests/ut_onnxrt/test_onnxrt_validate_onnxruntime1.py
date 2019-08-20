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
from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate.validate import sklearn_operators, enumerate_validated_operator_opsets, summary_report


class TestOnnxrtValidateOnnxRuntime1(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_all_onnxruntime1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_all_onnxruntime1")
        if False:  # pylint: disable=W0125
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"GradientBoostingRegressor"}, opset_min=11, fLOG=fLOG,
                runtime='onnxruntime1', debug=True))
        else:
            rows = []
            for row in enumerate_validated_operator_opsets(
                    verbose, debug=None, fLOG=fLOG, runtime='onnxruntime1',
                    dump_folder=temp, skip_models={"GaussianProcessRegressor"}):
                rows.append(row)
                if len(rows) > 20:
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
