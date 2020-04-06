"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import train_test_split
import skl2onnx
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.validate import sklearn_operators, enumerate_validated_operator_opsets
from mlprodict.onnxrt.validate.validate_problems import _problems
from mlprodict.onnxrt.validate.validate_difference import measure_relative_difference


class TestRtValidateGradientBoosting(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_GradientBoostingRegressor1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GradientBoostingRegressor"}, opset_min=10, fLOG=fLOG,
            runtime='python', debug=False))
        self.assertGreater(len(rows), 1)
        max_diff = max(_.get('max_rel_diff_batch', 1e-11) for _ in rows)
        self.assertLesser(max_diff, 1e-2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_GradientBoostingClassifier(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GradientBoostingClassifier"}, opset_min=10, fLOG=fLOG,
            runtime='python', debug=True,
            filter_exp=lambda m, p: 'm-cl' in p))
        self.assertGreater(len(rows), 1)
        max_diff = max(_.get('max_rel_diff_batch', 1e-11) for _ in rows)
        self.assertLesser(max_diff, 1e-5)

    def test_validate_GradientBoostingClassifier_custom(self):
        mcl = _problems['m-cl']()
        (X, y, init_types, _, __, ___) = mcl
        X_train, X_test, y_train, _ = train_test_split(
            X, y, shuffle=True, random_state=2)
        cl = GradientBoostingClassifier(n_estimators=20)
        cl.fit(X_train, y_train)
        pred_skl = cl.predict_proba(X_test)

        model_onnx = to_onnx(cl, init_types[0][1])
        oinf = OnnxInference(model_onnx, runtime='python')
        pred_onx = oinf.run({'X': X_test.astype(numpy.float32)})
        diff = numpy.max(
            numpy.abs(pred_skl - pred_onx['output_probability'].values).ravel())
        if diff >= 1e-5:
            dd = [(numpy.max(numpy.abs(a - b)), i)
                  for i, (a, b) in enumerate(zip(pred_skl, pred_onx['output_probability'].values))]
            dd.sort(reverse=True)
            diff1 = dd[0][0]
            diff2 = dd[3][0]
            self.assertGreater(diff1, diff2)
            self.assertLesser(diff2, 1e-5)
        diff = measure_relative_difference(
            pred_skl, pred_onx['output_probability'])
        self.assertLesser(diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
