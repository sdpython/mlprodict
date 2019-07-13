"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestOnnxrtValidateNodeTime(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_node_time(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LogisticRegression"}, opset_min=10,
            debug=False, node_time=True, fLOG=fLOG))
        self.assertNotEmpty(rows)
        for row in rows:
            self.assertIn('bench-batch', row)
            self.assertIsInstance(row['bench-batch'], list)
            self.assertNotEmpty(row['bench-batch'])


if __name__ == "__main__":
    unittest.main()
