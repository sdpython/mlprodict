"""
@brief      test log(time=5s)
"""
import os
import unittest
import pickle
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import (
    get_temp_folder, ExtTestCase, skipif_circleci
)
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx


class TestOnnxrtValidateDumpAll(ExtTestCase):

    @skipif_circleci("too long")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_dump_all(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_dump_all")
        self.assertRaise(lambda: list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeClassifier"},
            filter_exp=lambda m, p: '64' not in p,
            fLOG=fLOG, dump_all=True)),
            ValueError)
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"DecisionTreeClassifier"},
            filter_exp=lambda m, p: '64' not in p,
            fLOG=fLOG, dump_all=True,
            dump_folder=temp))
        self.assertGreater(len(rows), 1)
        df = DataFrame(rows)
        self.assertGreater(df.shape[1], 1)
        fLOG("output results")
        df.to_csv(os.path.join(temp, "sklearn_opsets_report.csv"), index=False)
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)

        stored = os.path.join(
            temp, ("dump-i-python-DecisionTreeClassifier-default-b-cl-tree._classes."
                   "DecisionTreeClassifierzipmapFalse-op%d-nf4.pkl" % get_opset_number_from_onnx()))
        with open(stored, "rb") as f:
            obj = pickle.load(f)
        self.assertIn('onnx_bytes', obj)
        self.assertIn('skl_model', obj)
        self.assertIn('X_test', obj)
        self.assertIn('Xort_test', obj)


if __name__ == "__main__":
    unittest.main()
