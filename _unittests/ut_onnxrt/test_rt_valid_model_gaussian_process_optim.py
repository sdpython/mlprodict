"""
@brief      test log(time=9s)
"""
import unittest
from logging import getLogger
from pandas import DataFrame
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets, summary_report


class TestRtValidateGaussianProcessOptim(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_GaussianProcessRegressor_python_optim(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"GaussianProcessRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug,
            filter_scenario=lambda m, p, s, e: p == "b-reg" and s == "rbf"))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)
        opt = set(_.get('optim', '') for _ in rows)
        self.assertEqual(opt, {'', 'onnx-optim=cdist', 'optim=cdist', 'onnx'})
        piv = summary_report(DataFrame(rows))
        ops = set(piv['opset11'])
        opt = set(piv['optim'])
        self.assertEqual(opt, {'', 'onnx-optim=cdist', 'optim=cdist', 'onnx'})
        self.assertGreater(len(ops), 1)


if __name__ == "__main__":
    unittest.main()
