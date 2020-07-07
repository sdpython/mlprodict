"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets
from mlprodict.onnxrt.validate.validate_problems import _modify_dimension


class TestRtValidateSVM(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_simple_test(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        for nf in range(16, 50):
            with self.subTest(nf=nf):
                iris = load_iris()
                X, y = iris.data, iris.target
                X = _modify_dimension(X, nf)
                X_train, X_test, y_train, _ = train_test_split(X, y)
                clr = SVR(kernel='linear')
                clr.fit(X_train, y_train)

                x2 = X_test.astype(numpy.float32)
                onx = to_onnx(clr, x2)
                pyrun = OnnxInference(onx, runtime="python")
                res = pyrun.run({'X': x2})
                self.assertIn('variable', res)
                self.assertEqual(res['variable'].shape, (38, ))
                self.assertEqualArray(
                    res['variable'], clr.predict(x2), decimal=2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_simple_test_double(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        for nf in range(16, 50):
            with self.subTest(nf=nf):
                iris = load_iris()
                X, y = iris.data, iris.target
                X = _modify_dimension(X, nf)
                X_train, X_test, y_train, _ = train_test_split(X, y)
                clr = SVR(kernel='linear')
                clr.fit(X_train, y_train)

                x2 = X_test.astype(numpy.float64)
                onx = to_onnx(clr, x2)
                pyrun = OnnxInference(onx, runtime="python")
                res = pyrun.run({'X': x2})
                self.assertIn('variable', res)
                self.assertEqual(res['variable'].shape, (38, ))
                self.assertEqualArray(
                    res['variable'], clr.predict(x2), decimal=2)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_python_rbf(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"SVR"},
            fLOG=myprint, benchmark=False,
            n_features=[45],
            runtime='python', debug=debug,
            filter_exp=lambda m, p: "64" not in p,
            filter_scenario=lambda m, p, s, e, t: "rbf" in str(e)))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_svr_python_linear(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"SVR"},
            fLOG=myprint, benchmark=False,
            n_features=[45],
            runtime='python', debug=debug,
            filter_exp=lambda m, p: "64" not in p,
            filter_scenario=lambda m, p, s, e, t: "linear" in str(e)))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
