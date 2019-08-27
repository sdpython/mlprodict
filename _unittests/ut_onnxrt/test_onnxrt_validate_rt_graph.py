"""
@brief      test log(time=5s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mlprodict.onnxrt.validate.validate import enumerate_validated_operator_opsets, summary_report
from mlprodict.onnxrt.validate.validate_graph import plot_validate_benchmark


class TestOnnxrtValidateRtGraph(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_validate_pyrt_ort(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"LinearRegression"}, opset_min=11, fLOG=fLOG,
            runtime=['python', 'onnxruntime1'], debug=False,
            benchmark=True, n_features=[None, 10]))

        df = DataFrame(rows)
        piv = summary_report(df)
        import matplotlib.pyplot as plt
        fig, ax = plot_validate_benchmark(piv)
        # plt.show()
        plt.clf()
        self.assertNotEmpty(fig)
        self.assertNotEmpty(ax)


if __name__ == "__main__":
    unittest.main()
