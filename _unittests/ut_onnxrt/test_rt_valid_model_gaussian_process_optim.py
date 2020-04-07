"""
@brief      test log(time=9s)
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
            verbose, models={"GaussianProcessRegressor"},
            fLOG=myprint,
            runtime='python', debug=debug,
            filter_scenario=lambda m, p, s, e, e2: p == "b-reg" and s == "rbf"))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)
        opt = set(_.get('optim', '') for _ in rows)
        expcl = "<class 'sklearn.gaussian_process.gpr.GaussianProcessRegressor'>={'optim': 'cdist'}"
        exp = [{'', 'onnx/' + expcl, expcl, 'onnx'},
               {'', 'onnx/' + expcl, expcl}]
        for i in range(len(exp)):  # pylint: disable=C0200
            exp[i] = set(e.replace("._gpr", ".gpr") for e in exp[i])
        opt = set(e.replace("._gpr", ".gpr") for e in opt)
        self.assertIn(opt, exp)
        piv = summary_report(DataFrame(rows))
        opt = set(piv['optim'])
        expcl = "cdist"
        exp = [{'', 'onnx/' + expcl, expcl, 'onnx'},
               {'', 'onnx/' + expcl, expcl}]
        self.assertIn(opt, exp)


if __name__ == "__main__":
    unittest.main()
