"""
@brief      test log(time=16s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, skipif_appveyor
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import (
    enumerate_validated_operator_opsets, summary_report
)


class TestRtValidateTfIdf(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_appveyor('stuck')
    def test_rt_tfidfvectorizer_onnxruntime1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        try:
            rows = list(enumerate_validated_operator_opsets(
                verbose, models={"TfidfVectorizer"},
                fLOG=myprint,
                runtime='onnxruntime1', debug=debug,
                filter_exp=lambda m, p: True))
        except Exception as e:
            if "Failed to construct locale" in str(e):
                return
            raise e
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[0])
        self.assertIn('onx_size', rows[-1])
        piv = summary_report(DataFrame(rows))
        self.assertGreater(piv.shape[0], 1)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_tfidfvectorizer_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"TfidfVectorizer"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, p: True))
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[0])
        self.assertIn('onx_size', rows[-1])
        piv = summary_report(DataFrame(rows))
        self.assertGreater(piv.shape[0], 1)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    @skipif_appveyor('stuck')
    def test_rt_tfidftransformer_onnxruntime1(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        debug = False
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"TfidfTransformer"},
            fLOG=myprint,
            runtime='onnxruntime1', debug=debug,
            filter_exp=lambda m, p: True))
        self.assertGreater(len(rows), 1)
        self.assertIn('skl_nop', rows[0])
        self.assertIn('onx_size', rows[-1])
        piv = summary_report(DataFrame(rows))
        self.assertGreater(piv.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
