"""
@brief      test log(time=16s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame, read_csv
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report


class TestOnnxrtValidateModelOneHotEncoder(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_one_hot_encoder(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(__file__, "temp_validate_one_hot_encoder")
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"OneHotEncoder"},
            filter_exp=lambda m, p: True,
            debug=False, fLOG=fLOG, dump_folder=temp,
            benchmark=True))
        self.assertGreater(len(rows), 1)
        df = DataFrame(rows)
        self.assertGreater(df.shape[1], 1)
        fLOG("output results")
        data = os.path.join(temp, "sklearn_opsets_report.csv")
        df.to_csv(data, index=False)

        df = read_csv(data)
        piv = summary_report(df)
        self.assertGreater(piv.shape[0], 1)
        self.assertGreater(piv.shape[1], 10)
        self.assertIn('OneHotEncoder', set(piv['name']))
        piv.to_csv(os.path.join(
            temp, "sklearn_opsets_summary.csv"), index=False)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)


if __name__ == "__main__":
    unittest.main()
