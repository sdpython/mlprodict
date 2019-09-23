"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestRtValidateKNeighborsRegressor(ExtTestCase):

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_KNeighborsRegressor_python(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        clr = KNeighborsRegressor()
        clr.fit(X_train, y_train)

        x2 = X_test.astype(numpy.float32)
        onx = to_onnx(clr, x2, rewrite_ops=True)
        pyrun = OnnxInference(onx, runtime="python")
        res = pyrun.run({'X': x2})
        self.assertIn('variable', res)
        self.assertEqual(res['variable'].shape, (38, ))

    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_KNeighborsRegressor_onnxruntime(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True

        iris = load_iris()
        X, y = iris.data, iris.target.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y)
        clr = KNeighborsRegressor()
        clr.fit(X_train, y_train)

        x2 = X_test.astype(numpy.float32)
        onx = to_onnx(clr, x2, rewrite_ops=True, target_opset=10)
        pyrun = OnnxInference(onx, runtime="onnxruntime1")
        res = pyrun.run({'X': x2}, fLOG=print, verbose=1)
        self.assertIn('variable', res)
        self.assertEqual(res['variable'].shape, (38, ))

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @ignore_warnings(category=(UserWarning, ConvergenceWarning, RuntimeWarning))
    def test_rt_KNeighborsRegressor_python_validate(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0

        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        debug = True
        rows = list(enumerate_validated_operator_opsets(
            verbose, models={"KNeighborsRegressor"}, opset_min=11, fLOG=myprint,
            runtime='python', debug=debug,
            filter_exp=lambda m, p: "b-reg" in p))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)


if __name__ == "__main__":
    unittest.main()
