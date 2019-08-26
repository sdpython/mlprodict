"""
@brief      test log(time=40s)
"""
import os
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame, read_csv
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import (
    get_temp_folder, ExtTestCase, skipif_circleci, unittest_require_at_least
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import skl2onnx
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
from mlprodict.onnxrt.validate.validate_problems import _modify_dimension


class TestOnnxrtValidate(ExtTestCase):

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @skipif_circleci("too long")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_all(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(__file__, "temp_validate_sklearn_operators_all")
        if False:  # pylint: disable=W0125
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"AdaBoostRegressor"}, opset_min=10,
                debug=True, fLOG=fLOG))
        else:
            rows = list(enumerate_validated_operator_opsets(
                verbose, debug=None, fLOG=fLOG, dump_folder=temp,
                opset_min=10, time_kwargs={10: dict(number=2, repeat=2)}))
        self.assertGreater(len(rows), 1)
        df = DataFrame(rows)
        self.assertGreater(df.shape[1], 1)
        fLOG("output results")
        df.to_csv(os.path.join(temp, "sklearn_opsets_report.csv"), index=False)
        df.to_excel(os.path.join(
            temp, "sklearn_opsets_report.xlsx"), index=False)

    def test_validate_summary(self):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "sklearn_opsets_report.csv")
        df = read_csv(data)
        piv = summary_report(df)
        self.assertGreater(piv.shape[0], 1)
        self.assertGreater(piv.shape[1], 10)
        self.assertIn('LogisticRegression', set(piv['name']))
        temp = get_temp_folder(__file__, "temp_validate_summary")
        fLOG("output results")
        piv.to_csv(os.path.join(
            temp, "sklearn_opsets_summary.csv"), index=False)
        piv.to_excel(os.path.join(
            temp, "sklearn_opsets_summary.xlsx"), index=False)

    def test_n_features_float(self):
        X = numpy.arange(20).reshape((5, 4)).astype(numpy.float64)
        X2 = _modify_dimension(X, 2)
        self.assertEqualArray(X[:, :2], X2)
        X2 = _modify_dimension(X, None)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 4)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 6)
        self.assertEqualArray(X[:, 2:4], X2[:, 2:4])
        self.assertNotEqualArray(X[:, :2], X2[:, :2])
        self.assertNotEqualArray(X[:, :2], X2[:, 4:6])
        cor = numpy.corrcoef(X2)
        for i in range(0, 2):
            cor = numpy.corrcoef(X[:, i], X2[:, i])
            self.assertLess(cor[0, 1], 0.9999)

    def test_n_features_int(self):
        X = numpy.arange(20).reshape((5, 4)).astype(numpy.int64)
        X2 = _modify_dimension(X, 2)
        self.assertEqualArray(X[:, :2], X2)
        X2 = _modify_dimension(X, None)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 4)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 6)
        self.assertEqualArray(X[:, 2:4], X2[:, 2:4])
        self.assertNotEqualArray(X[:, :2], X2[:, :2])
        self.assertNotEqualArray(X[:, :2], X2[:, 4:6])


if __name__ == "__main__":
    unittest.main()
