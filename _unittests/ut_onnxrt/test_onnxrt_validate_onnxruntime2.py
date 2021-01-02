"""
@brief      test log(time=10s)
"""
import os
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import (
    get_temp_folder, ExtTestCase, skipif_appveyor)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx


ignored_warnings = (UserWarning, ConvergenceWarning,
                    RuntimeWarning, FutureWarning)


class TestOnnxrtValidateOnnxRuntime(ExtTestCase):

    @skipif_appveyor('crashes')
    @ignore_warnings(category=ignored_warnings)
    def test_validate_sklearn_operators_onnxruntime_KMeans(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        op = get_opset_number_from_onnx()
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"KMeans"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=True,
            filter_exp=lambda m, p: '-64' not in p,
            opset_min=op, opset_max=op))
        self.assertGreater(len(rows), 1)
        # self.assertGreater(len(buffer), 1)

    @skipif_appveyor('crashes')
    @ignore_warnings(category=ignored_warnings)
    def test_validate_sklearn_operators_onnxruntime_BernoulliNB(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = False
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"BernoulliNB"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @skipif_appveyor('crashes')
    @ignore_warnings(category=ignored_warnings)
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
            verbose, models={"AdaBoostRegressor"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=debug,
            filter_exp=lambda m, p: "-64" not in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @skipif_appveyor('crashes')
    @ignore_warnings(category=ignored_warnings)
    def test_validate_sklearn_operators_onnxruntime_LogisticRegression(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LogisticRegression"},
            fLOG=myprint,
            runtime='onnxruntime2', debug=False,
            filter_exp=lambda m, p: '-64' not in p))
        self.assertGreater(len(rows), 1)
        # self.assertGreater(len(buffer), 1)

    @skipif_appveyor('crashes')
    @ignore_warnings(category=ignored_warnings)
    def test_validate_sklearn_operators_all_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        temp = get_temp_folder(
            __file__, "temp_validate_sklearn_operators_all_onnxruntime2")
        if False:  # pylint: disable=W0125
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"PLSCanonical"},
                fLOG=fLOG,
                runtime='onnxruntime2', debug=True))
        else:
            rows = []
            for row in enumerate_validated_operator_opsets(
                    verbose, debug=None, fLOG=fLOG, runtime='onnxruntime2',
                    benchmark=False, dump_folder=temp,
                    filter_exp=lambda m, s: m not in {AdaBoostRegressor,
                                                      GaussianProcessClassifier}):
                rows.append(row)
                if len(rows) > 30:
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
