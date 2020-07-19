"""
@brief      test log(time=5s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report
from mlprodict.plotting.plotting import plot_validate_benchmark


class TestOnnxrtValidateRtGraph(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_pyrt_ort(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LinearRegression"},
            fLOG=fLOG,
            runtime=['python'], debug=False,
            benchmark=True, n_features=[None, 10]))

        df = DataFrame(rows)
        piv = summary_report(df)
        import matplotlib.pyplot as plt
        fig, ax = plot_validate_benchmark(piv)
        # plt.show()
        plt.clf()
        self.assertNotEmpty(fig)
        self.assertNotEmpty(ax)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_pyrt_ort2(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 0 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LinearRegression"},
            fLOG=fLOG,
            runtime=['python', 'onnxruntime1'], debug=False,
            filter_exp=lambda m, p: '-64' not in p,
            benchmark=True, n_features=[None, 10]))

        df = DataFrame(rows)
        piv = summary_report(df)
        import matplotlib.pyplot as plt
        fig, ax = plot_validate_benchmark(piv)
        if __name__ == "__main__":
            plt.show()
        plt.clf()
        self.assertNotEmpty(fig)
        self.assertNotEmpty(ax)


if __name__ == "__main__":
    unittest.main()
