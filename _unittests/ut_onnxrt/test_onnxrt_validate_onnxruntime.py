"""
@brief      test log(time=10s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report


class TestOnnxrtValidateOnnxRuntime(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_onnxruntime_KMeans(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"KMeans"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime', debug=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_onnxruntime_BernoulliNB(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"BernoulliNB"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime', debug=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_onnxruntime_AdaBoostRegressor(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"AdaBoostRegressor"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_onnxruntime_LogisticRegression(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LogisticRegression"}, opset_min=11, fLOG=myprint,
            runtime='onnxruntime', debug=True))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_sklearn_operators_all_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_all_onnxruntime")
        if False:  # pylint: disable=W0125
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"LogisticRegression"}, opset_min=11, fLOG=fLOG,
                runtime='onnxruntime', debug=True))
        else:
            rows = []
            for row in enumerate_validated_operator_opsets(verbose, debug=None, fLOG=fLOG,
                                                           runtime='onnxruntime', dump_folder=temp):
                rows.append(row)
                if __name__ != "__main__" and len(rows) >= 30:
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
    TestOnnxrtValidateOnnxRuntime(
    ).test_validate_sklearn_operators_onnxruntime_AdaBoostRegressor()
    unittest.main()
